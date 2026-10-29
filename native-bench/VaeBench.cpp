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
#include <dirent.h>
#include "VaeBench.hpp"
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static long rss(){long total=0,resident=0;std::ifstream f("/proc/self/statm");f>>total>>resident;return resident*sysconf(_SC_PAGESIZE)/1024;}
static long available(){std::ifstream f("/proc/meminfo");std::string s;while(std::getline(f,s))if(s.rfind("MemAvailable:",0)==0)return std::stol(s.substr(13));return -1;}
static Json frequencies(){
  Json values=Json::object();const std::string root="/sys/devices/system/cpu/cpufreq/";DIR* dir=opendir(root.c_str());if(!dir)return values;
  while(auto* entry=readdir(dir)){std::string name=entry->d_name;if(name.rfind("policy",0)!=0)continue;
    long current=-1,maximum=-1;std::string cores;std::ifstream(root+name+"/scaling_cur_freq")>>current;std::ifstream(root+name+"/scaling_max_freq")>>maximum;
    std::ifstream list(root+name+"/related_cpus");std::getline(list,cores);values[name]={{"currentKHz",current},{"maximumKHz",maximum},{"relatedCpus",cores}};
  }closedir(dir);return values;
}
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
std::string LevoRunVaeBench(const std::string& dir,const std::string& precision,int threads,bool performanceCores){
  Json result={{"success",false},{"scope","Real pretrained VAE decoder only; synthetic latent stream; no DiT/LLM or speaker playback"},
               {"mnnVersion","3.6.1"},{"backend","CPU"},{"threadsRequested",threads},{"memoryMode","Low"},
               {"sampleIntervalMs",20},{"fullCodecValidated",false},{"speakerPlaybackTested",false}};
  Samples samples;auto start=Clock::now();
  try{
    if(precision!="high"&&precision!="low")throw std::runtime_error("Invalid precision");
    if(threads<1||threads>8)throw std::runtime_error("Invalid CPU thread count");
    result["powerHint"]=performanceCores?"High":"Normal";
    result["frequencyBeforeLoad"]=frequencies();
    Json manifest;std::ifstream fixture(dir+"/fixture.json");fixture>>manifest;
    const size_t frames=manifest.at("frames").get<size_t>(), overlapFrames=manifest.at("overlap_frames").get<size_t>();
    const int chunks=manifest.at("chunks").get<int>();
    if((frames!=64&&frames!=256)||overlapFrames!=frames/4||chunks<1||chunks>32||
       frames+(chunks-1)*(frames-overlapFrames)!=640)throw std::runtime_error("Unsupported fixed stream fixture");
    result["windowFrames"]=frames;result["chunks"]=chunks;
    if(available()<2*1024*1024)throw std::runtime_error("Less than2GiB system available; not started");
    result["precision"]=precision;result["rssBeforeKiB"]=rss();
    const std::string modelName="vae-"+std::to_string(frames)+"-fp32.mnn";
    auto tick=Clock::now();std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/"+modelName).c_str()));
    if(!net)throw std::runtime_error("Cannot read real decoder graph");
    std::string external=dir+"/"+modelName+".weight";net->setExternalFile(external.c_str());net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);result["readSeconds"]=elapsed(tick);
    MNN::BackendConfig backend;backend.precision=precision=="high"?MNN::BackendConfig::Precision_High:MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;
    backend.power=performanceCores?MNN::BackendConfig::Power_High:MNN::BackendConfig::Power_Normal;
    MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=threads;config.backendConfig=&backend;
    tick=Clock::now();auto* session=net->createSession(config);if(!session)throw std::runtime_error("Decoder session failed");result["createSeconds"]=elapsed(tick);
    tick=Clock::now();net->resizeSession(session);net->releaseModel();result["resizeSeconds"]=elapsed(tick);result["rssAfterResizeKiB"]=rss();samples.sample();
    int backends[2]={-1,-1};if(net->getSessionInfo(session,MNN::Interpreter::BACKENDS,backends))result["mainPipelineBackend"]=backends[0];
    constexpr size_t frameSamples=1920;const size_t full=frames*frameSamples,overlap=overlapFrames*frameSamples;
    std::vector<float> pcm(2*full),tail;size_t emittedSamples=0;double computeSeconds=0;
    Json rows=Json::array();
    for(int index=0;index<chunks;++index){
      if(available()<800*1024)throw std::runtime_error("System available below800MiB; stopped");
      auto latent=load(dir+"/latent-"+std::to_string(index)+".f32",64*frames);auto frequencyBefore=frequencies();
      tick=Clock::now();auto* input=net->getSessionInput(session,"latent");if(!input)throw std::runtime_error("Missing latent input");
      MNN::Tensor inputHost(input,MNN::Tensor::CAFFE,true);if(inputHost.elementSize()!=latent.size())throw std::runtime_error("Latent shape mismatch");
      std::memcpy(inputHost.host<float>(),latent.data(),latent.size()*4);if(!input->copyFromHostTensor(&inputHost))throw std::runtime_error("Latent copy failed");
      if(net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("VAE run failed");
      auto* output=net->getSessionOutput(session,"pcm");if(!output)throw std::runtime_error("Missing PCM output");
      MNN::Tensor host(output,MNN::Tensor::CAFFE,true);if(host.elementSize()!=pcm.size()||!output->copyToHostTensor(&host))throw std::runtime_error("PCM shape/copy failed");
      std::memcpy(pcm.data(),host.host<float>(),pcm.size()*4);double run=elapsed(tick);computeSeconds+=run;
      auto reference=load(dir+"/pcm-"+std::to_string(index)+".f32",2*full);double l2=relative(pcm,reference);
      if(index){if(tail.size()!=2*overlap)throw std::runtime_error("Missing previous PCM tail");for(size_t channel=0;channel<2;++channel)for(size_t i=0;i<overlap;++i){float alpha=float(i)/float(overlap-1);pcm[channel*full+i]=tail[channel*overlap+i]*(1-alpha)+pcm[channel*full+i]*alpha;}}
      size_t count=index==chunks-1?full:full-overlap;std::vector<float> piece(2*count);
      for(size_t channel=0;channel<2;++channel)std::copy_n(pcm.begin()+channel*full,count,piece.begin()+channel*count);
      if(index<chunks-1){tail.resize(2*overlap);for(size_t channel=0;channel<2;++channel)std::copy_n(pcm.begin()+channel*full+full-overlap,overlap,tail.begin()+channel*overlap);}else tail.clear();
      auto emittedReference=load(dir+"/emit-"+std::to_string(index)+".f32",2*count);double emittedL2=relative(piece,emittedReference);
      emittedSamples+=count;samples.sample();
      rows.push_back({{"chunk",index+1},{"runAndReadSeconds",run},{"relativeL2VsPytorch",l2},{"emittedRelativeL2",emittedL2},{"emittedSeconds",double(count)/48000},{"cumulativePcmSeconds",double(emittedSamples)/48000},{"wallSeconds",elapsed(start)},{"rssKiB",rss()},{"systemAvailableKiB",available()},{"frequencyBefore",frequencyBefore},{"frequencyAfter",frequencies()}});
      if(index==0)result["firstNumericalPacketSeconds"]=elapsed(start);
      std::cout<<"LEVO_VAE_PROGRESS "<<rows.back().dump()<<std::endl;
    }
    if(emittedSamples!=640*frameSamples)throw std::runtime_error("Duplicated or dropped PCM at stream end");
    result["rows"]=rows;result["decoderComputeSeconds"]=computeSeconds;result["validPcmSeconds"]=25.6;result["decoderOnlyComputeRtf"]=computeSeconds/25.6;
    result["oneSessionReused"]=true;result["pcmAccumulatedAcrossWholeSong"]=false;result["retainedOverlapFrames"]=overlapFrames;
    net->releaseSession(session);net.reset();result["rssAfterReleaseKiB"]=rss();result["success"]=true;
  }catch(const std::exception& error){result["error"]=error.what();}
  samples.sample();result["sampledPeakRssKiB"]=samples.peak.load();result["minimumSystemAvailableKiB"]=samples.minimum.load();result["totalSeconds"]=elapsed(start);
  return result.dump();
}
#ifndef LEVO_VAE_LIBRARY
int main(int argc,char** argv){
  if(argc<3||argc>5){std::cerr<<"Usage: levo_vae_bench fixtureDir high|low [threads] [normal|high]\n";return 2;}
  try{
    int threads=argc>3?std::stoi(argv[3]):4;std::string power=argc>4?argv[4]:"normal";
    if(power!="normal"&&power!="high")throw std::runtime_error("Invalid power hint");
    auto result=LevoRunVaeBench(argv[1],argv[2],threads,power=="high");std::cout<<"LEVO_VAE_REPORT "<<result<<std::endl;
    return Json::parse(result)["success"].get<bool>()?0:1;
  }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 2;}
}
#endif
