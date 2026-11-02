#pragma once
#include <filesystem>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <llm/llm.hpp>
#include <json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#ifdef LEVO_VAE_VECTOR_SIN
#include "kernels/VectorSin.hpp"
#include "kernels/SnakeRuntime.hpp"
#endif
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using MNN::Transformer::Llm;
inline std::atomic<void(*)(const char*)> levoStageHook{nullptr};
static double seconds(Clock::time_point tick){return std::chrono::duration<double>(Clock::now()-tick).count();}
static long resident(){long a=0,b=0;std::ifstream("/proc/self/statm")>>a>>b;return b*sysconf(_SC_PAGESIZE)/1024;}
static long available(){std::ifstream in("/proc/meminfo");std::string line;while(std::getline(in,line))if(line.rfind("MemAvailable:",0)==0)return std::stol(line.substr(13));return -1;}
static void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
static Json readJson(const std::string& path){std::ifstream in(path);require(in.good(),"Missing JSON: "+path);Json value;in>>value;return value;}
static std::vector<float> read(const std::string& path,size_t count){std::ifstream in(path,std::ios::binary|std::ios::ate);require(in&&in.tellg()==std::streamoff(count*4),"Fixture size/read mismatch: "+path);std::vector<float> value(count);in.seekg(0);in.read(reinterpret_cast<char*>(value.data()),count*4);require(in.good(),"Short fixture read");return value;}
static double peak(const std::vector<float>& values){double result=0;for(float v:values){require(std::isfinite(v),"Non-finite tensor");result=std::max(result,std::abs(double(v)));}return result;}
static double relative(const std::vector<float>& a,const std::vector<float>& b){require(a.size()==b.size(),"Reference shape mismatch");double error=0,energy=0;for(size_t i=0;i<a.size();++i){double d=a[i]-b[i];error+=d*d;energy+=double(b[i])*b[i];}return std::sqrt(error/std::max(energy,1e-30));}
struct Trace{
  Clock::time_point start=Clock::now();std::atomic<bool> stop{false};std::atomic<long> maxRss{0},minAvailable{LONG_MAX};std::thread sampler;Json stages=Json::array();std::string sink;
  Trace(){sampler=std::thread([this]{while(!stop){sample();std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}
  ~Trace(){stop=true;sampler.join();}
  void sample(){long value=resident(),old=maxRss.load();while(value>old&&!maxRss.compare_exchange_weak(old,value)){}value=available();old=minAvailable.load();if(value>=0)while(value<old&&!minAvailable.compare_exchange_weak(old,value)){};}
  void mark(const std::string& name){if(auto hook=levoStageHook.load())hook(name.c_str());sample();Json row={{"stage",name},{"seconds",seconds(start)},{"rssKiB",resident()},{"systemAvailableKiB",available()}};stages.push_back(row);if(!sink.empty()){std::ofstream out(sink,std::ios::app);out<<row.dump()<<'\n';}std::cout<<"LEVO_CHAIN_PROGRESS "<<row.dump()<<std::endl;}
  void gate(long mib=1024){sample();require(available()>=mib*1024,"System available below stage safety gate");require(resident()<5L*1024*1024,"Process RSS reached5GiB diagnostic stop");}
};
class Session{
  std::unique_ptr<MNN::Interpreter> net;MNN::Session* session=nullptr;
#ifdef LEVO_VAE_VECTOR_SIN
  std::unique_ptr<LevoVaeSnake::Runtime> snake;
  bool snakeValidated=false;double validationSeconds=0,validationL2=0;
#endif
public:
  Session(const std::string& model,bool low,int threads=6,bool snakeFusion=false,bool vectorSin=false){
    net.reset(MNN::Interpreter::createFromFile(model.c_str()));require(bool(net),"Cannot read MNN model");
    if(snakeFusion){require(low&&threads==6,"Snake runtime requires VAE CPU Low / 6 threads");net->setSessionMode(MNN::Interpreter::Session_Debug);}
    auto weight=model+".weight";net->setExternalFile(weight.c_str());net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
    MNN::BackendConfig backend;backend.precision=low?MNN::BackendConfig::Precision_Low:MNN::BackendConfig::Precision_High;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig cfg;cfg.type=MNN_FORWARD_CPU;cfg.numThread=threads;cfg.backendConfig=&backend;
    session=net->createSession(cfg);require(session,"Cannot create MNN session");
#ifdef LEVO_VAE_VECTOR_SIN
    // A VAE-only fresh process has no previous runtime to initialize ARM82.
    // Hook after backend creation and before the deferred graph resize.
    LevoVaeSin::Scope sinScope(vectorSin);
#else
    require(!vectorSin,"Vector sine not compiled");
#endif
    net->resizeSession(session);net->releaseModel();
#ifdef LEVO_VAE_VECTOR_SIN
    if(snakeFusion)snake=std::make_unique<LevoVaeSnake::Runtime>(model,model.substr(0,model.find_last_of("/\\")+1)+"snake-coefficients.json");
#else
    require(!snakeFusion,"Snake runtime not compiled");
#endif
  }
  ~Session(){if(session)net->releaseSession(session);}
  std::vector<float> run(const std::map<std::string,std::vector<float>>& values,const std::string& output,size_t count){
    auto putInputs=[&](){for(const auto& pair:values){auto* input=net->getSessionInput(session,pair.first.c_str());require(input,"Missing model input: "+pair.first);MNN::Tensor host(input,MNN::Tensor::CAFFE,true);require(host.elementSize()==pair.second.size(),"Model input size mismatch");std::memcpy(host.host<float>(),pair.second.data(),pair.second.size()*4);require(input->copyFromHostTensor(&host),"Input copy failed");}};putInputs();
    std::vector<float> reference;
#ifdef LEVO_VAE_VECTOR_SIN
    // One real-input reference forward in this same VAE session validates App
    // integration. Its full cost is included in the first chunk and total RTF.
    if(snake&&!snakeValidated){auto tick=Clock::now();require(net->runSession(session)==MNN::NO_ERROR,"Snake reference forward failed");auto* tensor=net->getSessionOutput(session,output.c_str());require(tensor,"Missing reference PCM");MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(host.elementSize()==count&&tensor->copyToHostTensor(&host),"Snake reference copy failed");reference.assign(host.host<float>(),host.host<float>()+count);peak(reference);putInputs();validationSeconds=seconds(tick);}
    if(snake)snake->run(net.get(),session);else
#endif
    require(net->runSession(session)==MNN::NO_ERROR,"MNN forward failed");auto* tensor=net->getSessionOutput(session,output.c_str());require(tensor,"Missing model output");MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(host.elementSize()==count&&tensor->copyToHostTensor(&host),"Output shape/copy failed");std::vector<float> result(count);std::memcpy(result.data(),host.host<float>(),count*4);peak(result);
#ifdef LEVO_VAE_VECTOR_SIN
    if(snake&&!snakeValidated){validationL2=relative(result,reference);require(validationL2<.001,"Snake real-input PCM parity failed; L2="+std::to_string(validationL2));snakeValidated=true;}
#endif
    return result;
  }
  Json snakeStats()const{
#ifdef LEVO_VAE_VECTOR_SIN
    if(snake){auto stats=snake->stats();stats["firstRealInputParityPassed"]=snakeValidated;stats["firstRealInputPcmL2"]=validationL2;stats["includedReferenceValidationSeconds"]=validationSeconds;return stats;}
#endif
    return Json();
  }
};
