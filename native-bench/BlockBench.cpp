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
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
long rssKiB(){long total=0,resident=0;std::ifstream f("/proc/self/statm");f>>total>>resident;return resident*sysconf(_SC_PAGESIZE)/1024;}
long availableKiB(){std::ifstream f("/proc/meminfo");std::string line;while(std::getline(f,line))if(line.rfind("MemAvailable:",0)==0)return std::stol(line.substr(13));return -1;}
struct Sampler{
  std::atomic<bool> stop{false};std::atomic<long> peak{0},minimumAvailable{LONG_MAX};std::thread worker;
  Sampler(){worker=std::thread([this]{while(!stop){sample();std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}
  void sample(){long n=rssKiB();long old=peak.load();while(old<n&&!peak.compare_exchange_weak(old,n)){};n=availableKiB();old=minimumAvailable.load();if(n>=0)while(old>n&&!minimumAvailable.compare_exchange_weak(old,n)){};}
  ~Sampler(){stop=true;worker.join();}
};
std::vector<float> load(const std::string& path,size_t count){
  std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(count*4))throw std::runtime_error("Bad fixture: "+path);
  std::vector<float> x(count);f.seekg(0);f.read(reinterpret_cast<char*>(x.data()),count*4);return x;
}
int main(int argc,char** argv){
  Json report={{"success",false},{"scope","Mate80 Pro HDC native CLI, one real pretrained DiT block, fixed32frames; not App/full-DiT/song RTF"},
               {"mnnVersion","3.6.1"},{"memoryMode","Low"},{"sampleIntervalMs",20},{"dedicatedGpuMemoryMeasured",false}};
  Sampler sampler;auto start=Clock::now();
  try{
    if(argc!=5)throw std::runtime_error("Usage: levo_block_bench fixtureDir modelFile high|low threads");
    std::string dir=argv[1],name=argv[2],precision=argv[3];int threads=std::stoi(argv[4]);
    if(name.find('/')!=std::string::npos||name.find("..")!=std::string::npos||(precision!="high"&&precision!="low")||threads<1||threads>8)throw std::runtime_error("Invalid benchmark options");
    if(availableKiB()<1024*1024)throw std::runtime_error("System available memory below 1GiB; no model loaded");
    report["model"]=name;report["precision"]=precision;report["threadsRequested"]=threads;report["rssBeforeKiB"]=rssKiB();
    Json manifest;std::ifstream manifestFile(dir+"/fixture.json");manifestFile>>manifest;
    int frames=manifest.at("x").at("shape").at(1).get<int>();
    if(frames!=32&&frames!=128&&frames!=256)throw std::runtime_error("Unsupported fixed fixture length");
    report["frames"]=frames;
    report["scope"]="Mate80 Pro HDC native CLI, one real pretrained DiT block, fixed"+std::to_string(frames)+"frames; not App/full-DiT/song RTF";
    auto x=load(dir+"/x.f32",frames*1200),mod=load(dir+"/modulation.f32",7200),mask=load(dir+"/mask.f32",frames*frames),reference=load(dir+"/reference.f32",frames*1200);
    auto tick=Clock::now();
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/"+name).c_str()));
    if(!net)throw std::runtime_error("Cannot read MNN graph");
    std::string weight=dir+"/"+name+".weight";net->setExternalFile(weight.c_str());
    report["readSeconds"]=seconds(tick);net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
    MNN::BackendConfig backend;backend.precision=precision=="high"?MNN::BackendConfig::Precision_High:MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;
    MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=threads;config.backendConfig=&backend;
    tick=Clock::now();auto* session=net->createSession(config);if(!session)throw std::runtime_error("Session creation failed");
    report["createSeconds"]=seconds(tick);tick=Clock::now();net->resizeSession(session);net->releaseModel();
    report["resizeSeconds"]=seconds(tick);report["rssAfterResizeKiB"]=rssKiB();sampler.sample();
    std::vector<double> runs;std::vector<float> output(reference.size());
    auto put=[&](const char* name,const std::vector<float>& data){auto* input=net->getSessionInput(session,name);if(!input)throw std::runtime_error("Missing graph input");MNN::Tensor host(input,MNN::Tensor::CAFFE,true);if(host.elementSize()!=data.size())throw std::runtime_error("Input shape mismatch");std::memcpy(host.host<float>(),data.data(),data.size()*4);if(!input->copyFromHostTensor(&host))throw std::runtime_error("Input copy failed");};
    for(int repeat=0;repeat<11;++repeat){
      tick=Clock::now();put("x",x);put("modulation",mod);put("mask",mask);
      if(net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("MNN run failed");
      auto* tensor=net->getSessionOutput(session,"output");if(!tensor)throw std::runtime_error("Missing output");
      MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);if(host.elementSize()!=output.size()||!tensor->copyToHostTensor(&host))throw std::runtime_error("Output shape/copy error");
      std::memcpy(output.data(),host.host<float>(),output.size()*4);runs.push_back(seconds(tick));sampler.sample();
      for(float value:output)if(!std::isfinite(value))throw std::runtime_error("Non-finite output at iteration "+std::to_string(repeat));
      if(availableKiB()<800*1024)throw std::runtime_error("Available memory below800MiB; stopped");
    }
    double delta=0,energy=0;for(size_t i=0;i<output.size();++i){if(!std::isfinite(output[i]))throw std::runtime_error("Non-finite output");double d=double(output[i])-reference[i];delta+=d*d;energy+=double(reference[i])*reference[i];}
    report["relativeL2VsPytorch"]=std::sqrt(delta/std::max(energy,1e-30));report["firstSeconds"]=runs[0];
    std::vector<double> warm(runs.begin()+1,runs.end());std::sort(warm.begin(),warm.end());report["warmMedianSeconds"]=(warm[4]+warm[5])*0.5;report["runsSeconds"]=runs;
    report["rssAfterRunKiB"]=rssKiB();net->releaseSession(session);net.reset();report["rssAfterReleaseKiB"]=rssKiB();report["success"]=true;
  }catch(const std::exception& e){report["error"]=e.what();}
  sampler.sample();report["sampledPeakRssKiB"]=sampler.peak.load();report["minimumSystemAvailableKiB"]=sampler.minimumAvailable.load();report["totalSeconds"]=seconds(start);
  std::cout<<"LEVO_REPORT "<<report.dump()<<std::endl;return report["success"].get<bool>()?0:1;
}
