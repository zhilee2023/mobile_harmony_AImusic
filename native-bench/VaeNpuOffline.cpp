#include <neural_network_runtime/neural_network_core.h>
#include <CANNKit/hiai_options.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "VaeVectorSin.hpp"
#include <json.hpp>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>
using Json=nlohmann::json;using Clock=std::chrono::steady_clock;
struct RunOptions{std::string performance="default";int repeats=4;};static RunOptions options;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
static long rss(){long a=0,b=0;std::ifstream("/proc/self/statm")>>a>>b;return b*sysconf(_SC_PAGESIZE)/1024;}
struct Samples{std::atomic<bool> stop{false};std::atomic<long> peak{0};std::thread worker;Samples(){worker=std::thread([this]{while(!stop){long v=rss(),old=peak.load();while(v>old&&!peak.compare_exchange_weak(old,v)){}std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}~Samples(){stop=true;worker.join();}};
static std::vector<float> read(const std::string& path,size_t count){std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f)&&f.tellg()==std::streamoff(count*4),"Fixture size mismatch");std::vector<float> data(count);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),count*4);check(f.good(),"Fixture read failed");return data;}
static Json numeric(const std::vector<float>& actual,const std::vector<float>& expected){double error=0,energy=0,peak=0;check(actual.size()==expected.size(),"Reference mismatch");for(size_t i=0;i<actual.size();++i){check(std::isfinite(actual[i]),"Nonfinite block output");double d=double(actual[i])-expected[i];error+=d*d;energy+=double(expected[i])*expected[i];peak=std::max(peak,std::abs(double(actual[i])));}return {{"relativeL2",std::sqrt(error/std::max(energy,1e-30))},{"outputPeak",peak},{"finite",true}};}
struct Api{
 void* hiai=nullptr;void* core=nullptr;
#define NAMES(F) F(OH_NNDevice_GetAllDevicesID) F(OH_NNDevice_GetName) F(OH_NNCompilation_ConstructWithOfflineModelBuffer) F(OH_NNCompilation_SetDevice) F(OH_NNCompilation_Build) F(OH_NNCompilation_Destroy) F(OH_NNExecutor_Construct) F(OH_NNExecutor_Destroy) F(OH_NNExecutor_GetInputCount) F(OH_NNExecutor_GetOutputCount) F(OH_NNExecutor_CreateInputTensorDesc) F(OH_NNExecutor_CreateOutputTensorDesc) F(OH_NNExecutor_RunSync) F(OH_NNTensorDesc_Destroy) F(OH_NNTensorDesc_GetDataType) F(OH_NNTensorDesc_GetShape) F(OH_NNTensorDesc_GetByteSize) F(OH_NNTensor_Create) F(OH_NNTensor_Destroy) F(OH_NNTensor_GetDataBuffer) F(HMS_HiAIOptions_SetModelDeviceOrder) F(HMS_HiAIOptions_SetFallbackMode)
#define DECLARE(name) decltype(&name) name=nullptr;
 NAMES(DECLARE)
#undef DECLARE
 Api(){hiai=dlopen("libhiai_foundation.so",RTLD_NOW|RTLD_GLOBAL);check(hiai,"CANN library unavailable");auto version=reinterpret_cast<const char*(*)()>(dlsym(hiai,"HMS_HiAI_GetVersion"));if(version)version();core=dlopen("libneural_network_core.so",RTLD_NOW|RTLD_LOCAL);check(core,"NN core unavailable");
#define LOAD(name) name=reinterpret_cast<decltype(name)>(dlsym(core,#name));if(!name)name=reinterpret_cast<decltype(name)>(dlsym(hiai,#name));check(name,"Missing " #name);
 NAMES(LOAD)
#undef LOAD
 }
 ~Api(){if(core)dlclose(core);if(hiai)dlclose(hiai);}
};
struct Resources{Api& a;OH_NNCompilation* compilation=nullptr;OH_NNExecutor* executor=nullptr;std::vector<NN_Tensor*> tensors;std::vector<NN_TensorDesc*> descriptions;explicit Resources(Api& api):a(api){}~Resources(){for(auto* t:tensors)a.OH_NNTensor_Destroy(&t);if(executor)a.OH_NNExecutor_Destroy(&executor);if(compilation)a.OH_NNCompilation_Destroy(&compilation);for(auto* d:descriptions)a.OH_NNTensorDesc_Destroy(&d);}};
static Json cpu(const std::string& dir,const std::string& id,const std::vector<float>& input,const std::vector<float>& reference,bool w8){
 Json row={{"success",false},{"backend","MNN CPU Low 6 threads NEON Sin"},{"weightBits",w8?8:32}};
 try{auto start=Clock::now();const auto model=dir+"/"+id+(w8?"-cpu-w8.mnn":"-cpu-fp32.mnn");std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model.c_str()));check(bool(net),"CPU model missing");net->setExternalFile((model+".weight").c_str());net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
 MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;auto* session=net->createSession(config);check(session,"CPU session failed");LevoVaeSin::Scope sine(true);net->resizeSession(session);sine.restore();net->releaseModel();row["setupSeconds"]=seconds(start);
 auto* x=net->getSessionInput(session,"x");auto* y=net->getSessionOutput(session,"y");check(x&&y,"CPU model IO missing");MNN::Tensor hx(x,MNN::Tensor::CAFFE,true),hy(y,MNN::Tensor::CAFFE,true);check(hx.elementSize()==input.size()&&hy.elementSize()==reference.size(),"CPU shape mismatch");std::vector<float> output(reference.size());
 for(int repeat=0;repeat<options.repeats;++repeat){start=Clock::now();std::memcpy(hx.host<float>(),input.data(),input.size()*4);check(x->copyFromHostTensor(&hx),"CPU input copy failed");check(net->runSession(session)==MNN::NO_ERROR,"CPU run failed");check(y->copyToHostTensor(&hy),"CPU output copy failed");std::memcpy(output.data(),hy.host<float>(),output.size()*4);row["runIncludingCopiesSeconds"].push_back(seconds(start));}
 row["numeric"]=numeric(output,reference);row["vectorSinKernelCalls"]=LevoVaeSin::calls.load();row["vectorSinCounterScope"]="Reset at this session's selector Scope construction";row["success"]=true;net->releaseSession(session);
 }catch(const std::exception& e){row["error"]=e.what();}return row;
}
static Json npu(Api& a,size_t device,const std::string& dir,const Json& item,const std::vector<float>& input,const std::vector<float>& reference){
 Json row={{"success",false},{"cpuFallbackAllowed",false},{"npuOnlyRequested",true},{"onlineSupportQueryUsed",false}};
 try{const std::string id=item["id"];std::ifstream file(dir+"/"+id+"-fp16.omc",std::ios::binary|std::ios::ate);check(bool(file)&&file.tellg()>0&&file.tellg()<16*1024*1024,"OMC file unavailable or unbounded");std::vector<char> bytes(size_t(file.tellg()));file.seekg(0);file.read(bytes.data(),bytes.size());Resources r(a);auto start=Clock::now();
 r.compilation=a.OH_NNCompilation_ConstructWithOfflineModelBuffer(bytes.data(),bytes.size());check(r.compilation,"Offline compilation allocation failed");check(a.OH_NNCompilation_SetDevice(r.compilation,device)==OH_NN_SUCCESS,"Bind HIAI_F failed");HiAI_ExecuteDevice npu=HIAI_EXECUTE_DEVICE_NPU;
 row["npuOnlyOptionCode"]=int(a.HMS_HiAIOptions_SetModelDeviceOrder(r.compilation,&npu,1));row["disableFallbackCode"]=int(a.HMS_HiAIOptions_SetFallbackMode(r.compilation,HIAI_FALLBACK_DISABLED));check(row["npuOnlyOptionCode"]==0&&row["disableFallbackCode"]==0,"NPU-only/no-fallback option rejected");
 row["performanceHint"]=options.performance;
 if(options.performance!="default"){
     auto setPerformance=reinterpret_cast<decltype(&OH_NNCompilation_SetPerformanceMode)>(dlsym(a.core,"OH_NNCompilation_SetPerformanceMode"));
     if(!setPerformance)setPerformance=reinterpret_cast<decltype(setPerformance)>(dlsym(a.hiai,"OH_NNCompilation_SetPerformanceMode"));
     check(setPerformance,"Performance mode API unavailable");
     row["performanceOptionCode"]=int(setPerformance(r.compilation,options.performance=="high"?OH_NN_PERFORMANCE_HIGH:OH_NN_PERFORMANCE_EXTREME));
     check(row["performanceOptionCode"]==0,"Requested performance mode rejected");
 }
 const auto built=a.OH_NNCompilation_Build(r.compilation);row["buildCode"]=int(built);check(built==OH_NN_SUCCESS,"NPU-only offline build rejected");r.executor=a.OH_NNExecutor_Construct(r.compilation);check(r.executor,"Offline executor failed");
 size_t ni=0,no=0;check(a.OH_NNExecutor_GetInputCount(r.executor,&ni)==OH_NN_SUCCESS&&a.OH_NNExecutor_GetOutputCount(r.executor,&no)==OH_NN_SUCCESS&&ni==1&&no==1,"OMC IO count mismatch");
 auto make=[&](bool output){auto* d=output?a.OH_NNExecutor_CreateOutputTensorDesc(r.executor,0):a.OH_NNExecutor_CreateInputTensorDesc(r.executor,0);check(d,"OMC descriptor missing");r.descriptions.push_back(d);OH_NN_DataType type=OH_NN_UNKNOWN;int32_t* shape=nullptr;size_t rank=0,size=0;a.OH_NNTensorDesc_GetDataType(d,&type);a.OH_NNTensorDesc_GetShape(d,&shape,&rank);a.OH_NNTensorDesc_GetByteSize(d,&size);check(type==OH_NN_FLOAT32&&shape&&std::vector<int32_t>(shape,shape+rank)==item["shape"].get<std::vector<int32_t>>()&&size==input.size()*4,"OMC FP32 interface shape/bytes mismatch");auto* t=a.OH_NNTensor_Create(device,d);check(t,"Device tensor allocation failed");r.tensors.push_back(t);return t;};
 auto* x=make(false);auto* y=make(true);auto* px=a.OH_NNTensor_GetDataBuffer(x);auto* py=a.OH_NNTensor_GetDataBuffer(y);check(px&&py,"Device tensor map failed");row["setupSeconds"]=seconds(start);row["rssAfterSetupKiB"]=rss();std::vector<float> output(reference.size());
 for(int repeat=0;repeat<options.repeats;++repeat){start=Clock::now();std::memcpy(px,input.data(),input.size()*4);const double upload=seconds(start);auto runStart=Clock::now();const auto run=a.OH_NNExecutor_RunSync(r.executor,&x,1,&y,1);const double compute=seconds(runStart);check(run==OH_NN_SUCCESS,"NPU-only RunSync failed");auto readStart=Clock::now();std::memcpy(output.data(),py,output.size()*4);const double download=seconds(readStart);row["runIncludingCopiesSeconds"].push_back(seconds(start));row["hostInputCopySeconds"].push_back(upload);row["runSyncSeconds"].push_back(compute);row["hostOutputCopySeconds"].push_back(download);}
 row["numeric"]=numeric(output,reference);row["npuOnlyBuildAndRunSucceeded"]=true;row["success"]=true;row["npuPrivateWorkspacePeakMeasured"]=false;
 }catch(const std::exception& e){row["error"]=e.what();}return row;
}
int main(int argc,char** argv){Samples samples;Json report={{"probeCompleted",false},{"scope","Real residual block, synthetic activations, offline OMC NPU-only vs MNN CPU; not complete VAE or whole-chain RTF"}};
 try{check(argc>=2&&argc<=4,"Usage: fixtureDir [default|high|extreme] [4..16 runs]");std::string dir=argv[1];if(argc>2)options.performance=argv[2];if(argc>3)options.repeats=std::stoi(argv[3]);check(options.performance=="default"||options.performance=="high"||options.performance=="extreme","Invalid NPU performance hint");check(options.repeats>=4&&options.repeats<=16,"Invalid bounded repeat count");report["performanceHint"]=options.performance;report["repeatCount"]=options.repeats;Json manifest;std::ifstream(dir+"/manifest.json")>>manifest;report["ablation"]=manifest.value("ablation",std::string());Api api;const size_t* devices=nullptr;uint32_t count=0;check(api.OH_NNDevice_GetAllDevicesID(&devices,&count)==OH_NN_SUCCESS,"Device enumeration failed");size_t device=0;bool found=false;for(uint32_t i=0;i<count;++i){const char* name=nullptr;api.OH_NNDevice_GetName(devices[i],&name);if(name&&std::string(name)=="HIAI_F"){device=devices[i];found=true;}}check(found,"HIAI_F missing");
 for(const auto& item:manifest["candidateCases"]){std::string id=item["id"];check(id=="residual-d1-h1"||id=="residual-d1-h4"||id=="residual-stack3-h4"||id=="residual-stack3-compact"||id=="pureconv6-h4"||id=="pureconv-k7-d1-h4"||id=="pureconv-k7-d3-h4"||id=="pureconv-k7-d9-h4","Unknown bounded case");size_t n=1;for(auto v:item["shape"])n*=v.get<size_t>();check(n<=size_t(64)*128*640,"Case exceeds bound");auto input=read(dir+"/"+id+"-input.f32",n),reference=read(dir+"/"+id+"-reference.f32",n);Json row={{"id",id},{"shape",item["shape"]}};std::cout<<"LEVO_OFFLINE_STAGE "<<id<<" NPU"<<std::endl;row["npu"]=npu(api,device,dir,item,input,reference);row["cpuFp32LowNeon"]=cpu(dir,id,input,reference,false);row["cpuW8LowNeon"]=cpu(dir,id,input,reference,true);report["cases"].push_back(row);std::cout<<"LEVO_OFFLINE_CASE "<<row.dump()<<std::endl;}
 report["probeCompleted"]=true;
 }catch(const std::exception& e){report["error"]=e.what();}report["sampledPeakRssKiB"]=samples.peak.load();report["sampleIntervalMs"]=20;std::cout<<"LEVO_OFFLINE_REPORT "<<report.dump()<<std::endl;return report["probeCompleted"].get<bool>()?0:1;
}
