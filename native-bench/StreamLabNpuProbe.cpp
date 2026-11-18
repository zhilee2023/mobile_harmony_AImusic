// Bounded portable-OM compatibility probe; independent of all installed Apps.
#include <neural_network_runtime/neural_network_core.h>
#include <CANNKit/hiai_options.h>
#include <json.hpp>
#include <dlfcn.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <filesystem>
#include <arm_neon.h>
using Json=nlohmann::json;using Clock=std::chrono::steady_clock;
namespace {
void require(bool ok,const std::string& error){if(!ok)throw std::runtime_error(error);}
double seconds(Clock::time_point tick){return std::chrono::duration<double>(Clock::now()-tick).count();}
long rss(){long a=0,b=0;std::ifstream("/proc/self/statm")>>a>>b;return b*sysconf(_SC_PAGESIZE)/1024;}
void transpose4(const float* source,float* target,int rows,int cols){
    require(rows%4==0&&cols%4==0,"Bridge dimensions require4-wide tiles");
    for(int r=0;r<rows;r+=4)for(int c=0;c<cols;c+=4){
        auto a=vtrnq_f32(vld1q_f32(source+size_t(r)*cols+c),vld1q_f32(source+size_t(r+1)*cols+c));
        auto b=vtrnq_f32(vld1q_f32(source+size_t(r+2)*cols+c),vld1q_f32(source+size_t(r+3)*cols+c));
        vst1q_f32(target+size_t(c)*rows+r,vcombine_f32(vget_low_f32(a.val[0]),vget_low_f32(b.val[0])));
        vst1q_f32(target+size_t(c+1)*rows+r,vcombine_f32(vget_low_f32(a.val[1]),vget_low_f32(b.val[1])));
        vst1q_f32(target+size_t(c+2)*rows+r,vcombine_f32(vget_high_f32(a.val[0]),vget_high_f32(b.val[0])));
        vst1q_f32(target+size_t(c+3)*rows+r,vcombine_f32(vget_high_f32(a.val[1]),vget_high_f32(b.val[1])));
    }
}
struct Samples {
    std::atomic<bool> stop{false};std::atomic<long> peak{0};std::thread worker;
    Samples(){worker=std::thread([this]{while(!stop){sample();std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}
    void sample(){long value=rss(),old=peak.load();while(value>old&&!peak.compare_exchange_weak(old,value)){};}
    ~Samples(){stop=true;worker.join();}
};
template<class T>std::vector<T> read(const std::string& path,size_t maximum){
    std::ifstream in(path,std::ios::binary|std::ios::ate);auto bytes=in.tellg();require(in&&bytes>0&&size_t(bytes)<=maximum&&size_t(bytes)%sizeof(T)==0,"Missing/unbounded fixture: "+path);
    std::vector<T> data(size_t(bytes)/sizeof(T));in.seekg(0);in.read(reinterpret_cast<char*>(data.data()),bytes);require(in.good(),"Short fixture read");return data;
}
struct Api {
    void *hiai=nullptr,*core=nullptr;
#define API_NAMES(F) F(OH_NNDevice_GetAllDevicesID) F(OH_NNDevice_GetName) F(OH_NNCompilation_ConstructWithOfflineModelBuffer) F(OH_NNCompilation_SetDevice) F(OH_NNCompilation_Build) F(OH_NNCompilation_Destroy) F(OH_NNExecutor_Construct) F(OH_NNExecutor_Destroy) F(OH_NNExecutor_GetInputCount) F(OH_NNExecutor_GetOutputCount) F(OH_NNExecutor_CreateInputTensorDesc) F(OH_NNExecutor_CreateOutputTensorDesc) F(OH_NNExecutor_RunSync) F(OH_NNTensorDesc_Destroy) F(OH_NNTensorDesc_GetDataType) F(OH_NNTensorDesc_GetShape) F(OH_NNTensorDesc_GetByteSize) F(OH_NNTensor_Create) F(OH_NNTensor_Destroy) F(OH_NNTensor_GetDataBuffer) F(HMS_HiAIOptions_SetModelDeviceOrder) F(HMS_HiAIOptions_SetOperatorDeviceOrder) F(HMS_HiAIOptions_SetFallbackMode)
#define DECLARE(name) decltype(&name) name=nullptr;
    API_NAMES(DECLARE)
#undef DECLARE
    Api(){
        hiai=dlopen("libhiai_foundation.so",RTLD_NOW|RTLD_GLOBAL);require(hiai,"Missing CANN foundation");
        auto version=reinterpret_cast<const char*(*)()>(dlsym(hiai,"HMS_HiAI_GetVersion"));if(version)version();
        core=dlopen("libneural_network_core.so",RTLD_NOW|RTLD_LOCAL);require(core,"Missing NN core");
#define LOAD(name) name=reinterpret_cast<decltype(name)>(dlsym(core,#name));if(!name)name=reinterpret_cast<decltype(name)>(dlsym(hiai,#name));require(name,"Missing " #name);
        API_NAMES(LOAD)
#undef LOAD
    }
    ~Api(){if(core)dlclose(core);if(hiai)dlclose(hiai);}
};
struct Resources {
    Api& api;OH_NNCompilation* compilation=nullptr;OH_NNExecutor* executor=nullptr;std::vector<NN_Tensor*> tensors;std::vector<NN_TensorDesc*> descriptions;
    explicit Resources(Api& a):api(a){}
    ~Resources(){for(auto t:tensors)api.OH_NNTensor_Destroy(&t);if(executor)api.OH_NNExecutor_Destroy(&executor);if(compilation)api.OH_NNCompilation_Destroy(&compilation);for(auto d:descriptions)api.OH_NNTensorDesc_Destroy(&d);}
};
}
int main(int argc,char** argv){
    if(argc<3||argc>5)return 2;bool profiling=false,withBridge=false;
    for(int i=3;i<argc;++i){std::string flag=argv[i];if(flag=="--profile")profiling=true;else if(flag=="--bridge")withBridge=true;else return 2;}
    std::string dir=argv[1],sink=argv[2];Json report={{"success",false},{"scope","Portable OM captured DiT subgraph; manifest defines boundary, not full DiT or whole-chain RTF"},{"processId",getpid()},{"profilingRequested",profiling},
        {"format","portable OM IR"},{"cases",Json::array()},{"hardwareKernelTraceCollected",false},{"cpuSequenceLayoutBridgeTimed",withBridge}};
    Samples samples;auto save=[&]{samples.sample();report["sampledPeakRssMiB"]=samples.peak.load()/1024.;report["memoryScope"]="Process RSS,20ms plus stage checkpoints; NPU private workspace not measured";std::ofstream(sink)<<report.dump(2);};
    try{
        if(withBridge){
            for(auto shape:std::vector<std::pair<int,int>>{{256,1200},{1200,256},{264,1200}}){
                const auto rows=shape.first,cols=shape.second;std::vector<float> source(rows*cols),actual(rows*cols);
                for(size_t i=0;i<source.size();++i)source[i]=float(int(i%8191)-4000);
                transpose4(source.data(),actual.data(),rows,cols);
                for(int r=0;r<rows;++r)for(int c=0;c<cols;++c)require(actual[size_t(c)*rows+r]==source[size_t(r)*cols+c],"NEON transpose disagrees with scalar index reference");
                report["bridgeSelfTests"].push_back({{"rows",rows},{"cols",cols},{"exact",true}});
            }
        }
        Json manifest;std::ifstream(dir+"/manifest.json")>>manifest;report["fixtureScope"]=manifest.at("scope");Api api;const size_t* devices=nullptr;uint32_t count=0;
        require(api.OH_NNDevice_GetAllDevicesID(&devices,&count)==0,"Enumerating NN devices failed");size_t device=0;bool found=false;
        for(uint32_t i=0;i<count;++i){const char* name=nullptr;api.OH_NNDevice_GetName(devices[i],&name);if(name)report["devices"].push_back(name);if(name&&std::string(name)=="HIAI_F"){device=devices[i];found=true;}}
        require(found,"CANN HIAI_F not registered");
        for(const auto& item:manifest["cases"]){
            Json row={{"id",item["id"]},{"success",false},{"phase","reading"},{"runs",Json::array()}};
            auto checkpoint=[&]{Json active=report;active["activeCase"]=row;std::ofstream(sink)<<active.dump(2);};
            try{
                auto buffer=read<unsigned char>(dir+"/"+item["model"].get<std::string>(),100*1024*1024);
                auto x=read<float>(dir+"/"+item["input"].get<std::string>(),16*1024*1024),expected=read<float>(dir+"/"+item["reference"].get<std::string>(),16*1024*1024);
                const int inputFrames=item.value("sequenceInputFrames",256),outputFrames=item.value("sequenceOutputFrames",256);
                require(inputFrames>0&&inputFrames<=512&&inputFrames%4==0&&outputFrames>0&&outputFrames<=512&&outputFrames%4==0,"Unsupported sequence bridge sizes");
                const bool bridge=withBridge&&item["inputShape"].get<std::vector<int>>()==std::vector<int>({2,1200,1,inputFrames});
                row["cpuBtcNchwBridgeTimed"]=bridge;std::vector<float> sequenceInput,sequenceReference;
                if(bridge){require(x.size()==size_t(2*1200*inputFrames)&&expected.size()==size_t(2*1200*outputFrames),"Bridge element count mismatch");sequenceInput.resize(x.size());sequenceReference.resize(expected.size());for(int b=0;b<2;++b){transpose4(x.data()+b*1200*inputFrames,sequenceInput.data()+b*1200*inputFrames,1200,inputFrames);transpose4(expected.data()+b*1200*outputFrames,sequenceReference.data()+b*1200*outputFrames,1200,outputFrames);}}
                const auto operatorNames=item.value("operators",std::vector<std::string>{});
                const std::string profileDir=dir+"/profile-"+item["id"].get<std::string>();
                Resources owned(api);auto tick=Clock::now();owned.compilation=api.OH_NNCompilation_ConstructWithOfflineModelBuffer(buffer.data(),buffer.size());require(owned.compilation,"Offline constructor failed");
                row["deviceOptionCode"]=int(api.OH_NNCompilation_SetDevice(owned.compilation,device));require(row["deviceOptionCode"]==0,"HIAI_F device bind rejected");
                HiAI_ExecuteDevice npu=HIAI_EXECUTE_DEVICE_NPU;const std::string placement=item["placement"];
                if(placement=="model")row["placementCode"]=int(api.HMS_HiAIOptions_SetModelDeviceOrder(owned.compilation,&npu,1));
                else {require(placement=="operators"&&!operatorNames.empty(),"Unknown/empty placement policy");Json codes=Json::array();for(const auto& name:operatorNames)codes.push_back(int(api.HMS_HiAIOptions_SetOperatorDeviceOrder(owned.compilation,name.c_str(),&npu,1)));row["operatorPlacementCodes"]=codes;for(auto code:codes)require(code==0,"Operator device placement rejected");}
                if(placement=="model")require(row["placementCode"]==0,"Model device placement rejected");
                row["placement"]=placement;row["fallbackOptionCode"]=int(api.HMS_HiAIOptions_SetFallbackMode(owned.compilation,HIAI_FALLBACK_DISABLED));require(row["fallbackOptionCode"]==0,"Disable fallback rejected");
                auto performance=reinterpret_cast<decltype(&OH_NNCompilation_SetPerformanceMode)>(dlsym(api.core,"OH_NNCompilation_SetPerformanceMode"));
                if(!performance)performance=reinterpret_cast<decltype(performance)>(dlsym(api.hiai,"OH_NNCompilation_SetPerformanceMode"));require(performance,"Performance mode API missing");
                row["performanceModeCode"]=int(performance(owned.compilation,OH_NN_PERFORMANCE_HIGH));require(row["performanceModeCode"]==0,"High performance request rejected");
                if(profiling){
                    std::filesystem::create_directories(profileDir);
                    auto enable=reinterpret_cast<decltype(&HMS_HiAIOptions_SetOmOptions)>(dlsym(api.hiai,"HMS_HiAIOptions_SetOmOptions"));require(enable,"Profiling API unavailable");
                    row["profilingOptionCode"]=int(enable(owned.compilation,HIAI_OM_TYPE_PROFILING,profileDir.c_str()));require(row["profilingOptionCode"]==0,"Profiling option rejected");
                }
                row["phase"]="building";checkpoint();row["buildCode"]=int(api.OH_NNCompilation_Build(owned.compilation));row["loadBuildSeconds"]=seconds(tick);require(row["buildCode"]==0,"Portable OM build rejected");
                owned.executor=api.OH_NNExecutor_Construct(owned.compilation);require(owned.executor,"No executor");size_t ni=0,no=0;api.OH_NNExecutor_GetInputCount(owned.executor,&ni);api.OH_NNExecutor_GetOutputCount(owned.executor,&no);require(ni==1&&no==1,"Unexpected IO count");
                auto make=[&](bool output){auto d=output?api.OH_NNExecutor_CreateOutputTensorDesc(owned.executor,0):api.OH_NNExecutor_CreateInputTensorDesc(owned.executor,0);require(d,"No IO descriptor");owned.descriptions.push_back(d);OH_NN_DataType type=OH_NN_UNKNOWN;size_t bytes=0,rank=0;int32_t* shape=nullptr;api.OH_NNTensorDesc_GetDataType(d,&type);api.OH_NNTensorDesc_GetByteSize(d,&bytes);api.OH_NNTensorDesc_GetShape(d,&shape,&rank);
                    const auto wanted=item[output?"outputShape":"inputShape"].get<std::vector<int32_t>>();auto actual=std::vector<int32_t>(shape,shape+rank);auto nonUnit=[](std::vector<int32_t> a){a.erase(std::remove(a.begin(),a.end(),1),a.end());return a;};
                    require(type==OH_NN_FLOAT32&&bytes==(output?expected.size():x.size())*4&&nonUnit(actual)==nonUnit(wanted),"FP32 shape/size/order contract mismatch");row[output?"outputShape":"inputShape"]=actual;
                    auto tensor=api.OH_NNTensor_Create(device,d);require(tensor,"Device tensor allocation failed");owned.tensors.push_back(tensor);return tensor;};
                auto in=make(false),out=make(true);auto ip=api.OH_NNTensor_GetDataBuffer(in),op=api.OH_NNTensor_GetDataBuffer(out);require(ip&&op,"Tensor mapping failed");std::vector<float> result(expected.size());
                for(int repeat=0;repeat<4;++repeat){row["phase"]="running";checkpoint();tick=Clock::now();if(bridge){for(int b=0;b<2;++b)transpose4(sequenceInput.data()+b*1200*inputFrames,static_cast<float*>(ip)+b*1200*inputFrames,inputFrames,1200);}else std::memcpy(ip,x.data(),x.size()*4);double upload=seconds(tick);tick=Clock::now();int code=api.OH_NNExecutor_RunSync(owned.executor,&in,1,&out,1);double run=seconds(tick);require(code==0,"RunSync failed");tick=Clock::now();if(bridge){for(int b=0;b<2;++b)transpose4(static_cast<const float*>(op)+b*1200*outputFrames,result.data()+b*1200*outputFrames,1200,outputFrames);}else std::memcpy(result.data(),op,result.size()*4);double download=seconds(tick);
                    row["runs"].push_back({{"mode",repeat?"warm":"first"},{"inputSeconds",upload},{"runSyncSeconds",run},{"outputSeconds",download},{"totalSeconds",upload+run+download}});}
                const auto& reference=bridge?sequenceReference:expected;double error=0,energy=0;for(size_t i=0;i<result.size();++i){require(std::isfinite(result[i]),"Non-finite output");error+=std::pow(double(result[i])-reference[i],2);energy+=double(reference[i])*reference[i];}
                row["relativeL2"]=std::sqrt(error/std::max(energy,1e-30));row["numericalBudget"]=.01;row["success"]=row["relativeL2"].get<double>()<=.01;row["rssAtEndMiB"]=rss()/1024.;samples.sample();row["phase"]="done";
            }catch(const std::exception& e){row["error"]=e.what();}
            report["cases"].push_back(row);save();std::cout<<"OM_COMPAT_CASE "<<row.dump()<<std::endl;
        }
        report["success"]=true;report["meaningOfSuccess"]="Diagnostic completed; inspect individual build/run/numerical results, not proof of exclusive NPU execution";
    }catch(const std::exception& e){report["error"]=e.what();}
    save();return report["success"]==true?0:1;
}
