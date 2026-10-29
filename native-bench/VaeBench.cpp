#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include "VaeBench.hpp"
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static long rss(){long total=0,resident=0;std::ifstream f("/proc/self/statm");f>>total>>resident;return resident*sysconf(_SC_PAGESIZE)/1024;}
static long available(){std::ifstream f("/proc/meminfo");std::string s;while(std::getline(f,s))if(s.rfind("MemAvailable:",0)==0)return std::stol(s.substr(13));return -1;}
struct Samples{
  std::atomic<bool> stop{false};std::atomic<long> peak{0},minimum{LONG_MAX};std::thread worker;
  Samples(){worker=std::thread([this]{while(!stop){sample();std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}
  void sample(){long n=rss(),old=peak.load();while(old<n&&!peak.compare_exchange_weak(old,n)){}n=available();old=minimum.load();if(n>=0)while(old>n&&!minimum.compare_exchange_weak(old,n)){};}
  ~Samples(){stop=true;worker.join();}
};
static std::vector<float> load(const std::string& path,size_t count){
  std::ifstream file(path,std::ios::binary|std::ios::ate);
  if(!file||file.tellg()!=std::streamoff(count*4))throw std::runtime_error("Bad fixture: "+path);
  std::vector<float> values(count);file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),count*4);return values;
}
static double relative(const std::vector<float>& a,const std::vector<float>& b){
  if(a.size()!=b.size())throw std::runtime_error("Numerical sink length mismatch");
  double error=0,energy=0;for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i]))throw std::runtime_error("Non-finite decoder PCM");double d=double(a[i])-b[i];error+=d*d;energy+=double(b[i])*b[i];}return std::sqrt(error/std::max(energy,1e-30));
}
std::string LevoRunVaeBench(const std::string& dir,const std::string& precision){
  Json result={{"success",false},{"scope","Real pretrained VAE decoder only; synthetic latent stream; no DiT/LLM or speaker playback"},
               {"mnnVersion","3.6.1"},{"backend","CPU"},{"threadsRequested",4},{"memoryMode","Low"},
               {"sampleIntervalMs",20},{"fullCodecValidated",false},{"speakerPlaybackTested",false}};
  Samples samples;auto start=Clock::now();
  try{
    if(precision!="high"&&precision!="low")throw std::runtime_error("Invalid precision");
    Json manifest;std::ifstream fixture(dir+"/fixture.json");fixture>>manifest;
    if(manifest.at("frames")!=256||manifest.at("chunks")!=3||manifest.at("overlap_frames")!=64)throw std::runtime_error("Unsupported fixed stream fixture");
    if(available()<2*1024*1024)throw std::runtime_error("Less than2GiB system available; not started");
    result["precision"]=precision;result["rssBeforeKiB"]=rss();
    auto tick=Clock::now();std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/vae-256-fp32.mnn").c_str()));
    if(!net)throw std::runtime_error("Cannot read real decoder graph");
    std::string external=dir+"/vae-256-fp32.mnn.weight";net->setExternalFile(external.c_str());net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);result["readSeconds"]=elapsed(tick);
    MNN::BackendConfig backend;backend.precision=precision=="high"?MNN::BackendConfig::Precision_High:MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;
    MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=4;config.backendConfig=&backend;
    tick=Clock::now();auto* session=net->createSession(config);if(!session)throw std::runtime_error("Decoder session failed");result["createSeconds"]=elapsed(tick);
    tick=Clock::now();net->resizeSession(session);net->releaseModel();result["resizeSeconds"]=elapsed(tick);result["rssAfterResizeKiB"]=rss();samples.sample();
    constexpr size_t frameSamples=1920,full=256*frameSamples,overlap=64*frameSamples;
    std::vector<float> pcm(2*full),tail;size_t emittedSamples=0;double computeSeconds=0;
    Json rows=Json::array();
    for(int index=0;index<3;++index){
      if(available()<800*1024)throw std::runtime_error("System available below800MiB; stopped");
      auto latent=load(dir+"/latent-"+std::to_string(index)+".f32",64*256);
      tick=Clock::now();auto* input=net->getSessionInput(session,"latent");if(!input)throw std::runtime_error("Missing latent input");
      MNN::Tensor inputHost(input,MNN::Tensor::CAFFE,true);if(inputHost.elementSize()!=latent.size())throw std::runtime_error("Latent shape mismatch");
      std::memcpy(inputHost.host<float>(),latent.data(),latent.size()*4);if(!input->copyFromHostTensor(&inputHost))throw std::runtime_error("Latent copy failed");
      if(net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("VAE run failed");
      auto* output=net->getSessionOutput(session,"pcm");if(!output)throw std::runtime_error("Missing PCM output");
      MNN::Tensor host(output,MNN::Tensor::CAFFE,true);if(host.elementSize()!=pcm.size()||!output->copyToHostTensor(&host))throw std::runtime_error("PCM shape/copy failed");
      std::memcpy(pcm.data(),host.host<float>(),pcm.size()*4);double run=elapsed(tick);computeSeconds+=run;
      auto reference=load(dir+"/pcm-"+std::to_string(index)+".f32",2*full);double l2=relative(pcm,reference);
      if(index){if(tail.size()!=2*overlap)throw std::runtime_error("Missing previous PCM tail");for(size_t channel=0;channel<2;++channel)for(size_t i=0;i<overlap;++i){float alpha=float(i)/float(overlap-1);pcm[channel*full+i]=tail[channel*overlap+i]*(1-alpha)+pcm[channel*full+i]*alpha;}}
      size_t count=index==2?full:full-overlap;std::vector<float> piece(2*count);
      for(size_t channel=0;channel<2;++channel)std::copy_n(pcm.begin()+channel*full,count,piece.begin()+channel*count);
      if(index<2){tail.resize(2*overlap);for(size_t channel=0;channel<2;++channel)std::copy_n(pcm.begin()+channel*full+full-overlap,overlap,tail.begin()+channel*overlap);}else tail.clear();
      auto emittedReference=load(dir+"/emit-"+std::to_string(index)+".f32",2*count);double emittedL2=relative(piece,emittedReference);
      emittedSamples+=count;samples.sample();
      rows.push_back({{"chunk",index+1},{"runAndReadSeconds",run},{"relativeL2VsPytorch",l2},{"emittedRelativeL2",emittedL2},{"emittedSeconds",double(count)/48000},{"cumulativePcmSeconds",double(emittedSamples)/48000},{"wallSeconds",elapsed(start)},{"rssKiB",rss()},{"systemAvailableKiB",available()}});
      if(index==0)result["firstNumericalPacketSeconds"]=elapsed(start);
      std::cout<<"LEVO_VAE_PROGRESS "<<rows.back().dump()<<std::endl;
    }
    if(emittedSamples!=640*frameSamples)throw std::runtime_error("Duplicated or dropped PCM at stream end");
    result["rows"]=rows;result["decoderComputeSeconds"]=computeSeconds;result["validPcmSeconds"]=25.6;result["decoderOnlyComputeRtf"]=computeSeconds/25.6;
    result["oneSessionReused"]=true;result["pcmAccumulatedAcrossWholeSong"]=false;result["retainedOverlapFrames"]=64;
    net->releaseSession(session);net.reset();result["rssAfterReleaseKiB"]=rss();result["success"]=true;
  }catch(const std::exception& error){result["error"]=error.what();}
  samples.sample();result["sampledPeakRssKiB"]=samples.peak.load();result["minimumSystemAvailableKiB"]=samples.minimum.load();result["totalSeconds"]=elapsed(start);
  return result.dump();
}
#ifndef LEVO_VAE_LIBRARY
int main(int argc,char** argv){
  if(argc!=3){std::cerr<<"Usage: levo_vae_bench fixtureDir high|low\n";return 2;}
  auto result=LevoRunVaeBench(argv[1],argv[2]);std::cout<<"LEVO_VAE_REPORT "<<result<<std::endl;
  return Json::parse(result)["success"].get<bool>()?0:1;
}
#endif
