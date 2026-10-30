// Bounded online-graph capability/compile probe. Never runs inference and never
// treats SDK enums or an empty support list as actual NPU execution evidence.
#include <neural_network_runtime/neural_network_runtime.h>
#include <CANNKit/hiai_options.h>
#include <json.hpp>
#include <dlfcn.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
using Json=nlohmann::json;
static void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Api {
    void* core=nullptr;void* hiai=nullptr;void* runtime=nullptr;
#define NAMES(F) F(OH_NNDevice_GetAllDevicesID) F(OH_NNDevice_GetName) F(OH_NNModel_Construct) F(OH_NNModel_Destroy) F(OH_NNModel_AddTensorToModel) F(OH_NNModel_AddOperation) F(OH_NNModel_SpecifyInputsAndOutputs) F(OH_NNModel_Finish) F(OH_NNModel_GetAvailableOperations) F(OH_NNTensorDesc_Create) F(OH_NNTensorDesc_Destroy) F(OH_NNTensorDesc_SetDataType) F(OH_NNTensorDesc_SetShape) F(OH_NNTensorDesc_SetFormat) F(OH_NNCompilation_Construct) F(OH_NNCompilation_Destroy) F(OH_NNCompilation_SetDevice) F(OH_NNCompilation_Build) F(HMS_HiAIOptions_SetModelDeviceOrder) F(HMS_HiAIOptions_SetFallbackMode)
#define DECLARE(name) decltype(&name) name=nullptr;
    NAMES(DECLARE)
#undef DECLARE
    Api(){core=dlopen("libneural_network_core.so",RTLD_NOW|RTLD_LOCAL);runtime=dlopen("libneural_network_runtime.so",RTLD_NOW|RTLD_LOCAL);hiai=dlopen("libhiai_foundation.so",RTLD_NOW|RTLD_LOCAL);check(core&&hiai,"Public NNRT/CANN runtime unavailable");
#define LOAD(name) name=reinterpret_cast<decltype(name)>(dlsym(core,#name));if(!name&&runtime)name=reinterpret_cast<decltype(name)>(dlsym(runtime,#name));if(!name)name=reinterpret_cast<decltype(name)>(dlsym(hiai,#name));check(name,"Missing public API " #name);
        NAMES(LOAD)
#undef LOAD
    }
    ~Api(){if(hiai)dlclose(hiai);if(runtime)dlclose(runtime);if(core)dlclose(core);}
};
struct Resources {
    Api& api;OH_NNModel* model=nullptr;OH_NNCompilation* compilation=nullptr;std::vector<NN_TensorDesc*> descriptors;
    explicit Resources(Api& value):api(value){}
    ~Resources(){if(compilation)api.OH_NNCompilation_Destroy(&compilation);if(model)api.OH_NNModel_Destroy(&model);for(auto* d:descriptors)api.OH_NNTensorDesc_Destroy(&d);}
};
int main(int argc,char** argv){
    if(argc!=2||std::string(argv[1])!="--query-support"){
        std::cout<<"Explicit --query-support required. This legacy online support query SIGBUSed on the recorded device; no App or inference code uses it."<<std::endl;return 2;
    }
    Json report={{"probeCompleted",false},{"inferenceExecuted",false},{"scope","Sin online graph support/compile only, NPU-only requested, CPU fallback disabled; not VAE throughput"}};
    try{Api api;const size_t* devices=nullptr;uint32_t count=0;check(api.OH_NNDevice_GetAllDevicesID(&devices,&count)==OH_NN_SUCCESS,"Cannot enumerate devices");size_t selected=0;bool found=false;
        for(uint32_t i=0;i<count;++i){const char* name=nullptr;if(api.OH_NNDevice_GetName(devices[i],&name)!=OH_NN_SUCCESS||!name)continue;report["devices"].push_back(name);if(std::string(name)=="HIAI_F"){selected=devices[i];found=true;}}
        check(found,"HIAI_F device not exposed");
        for(auto type:{OH_NN_FLOAT16,OH_NN_FLOAT32}){
            Json row={{"op","Sin"},{"precision",type==OH_NN_FLOAT16?"FP16":"FP32"},{"shape",{1,128,1,8320}},{"onlineNpuBuildSucceeded",false},{"physicalPlacementVerified",false},{"cpuFallbackAllowed",false}};
            try{Resources r(api);r.model=api.OH_NNModel_Construct();check(r.model,"Model allocation failed");std::vector<int32_t> shape{1,128,1,8320};
                for(int i=0;i<2;++i){auto* d=api.OH_NNTensorDesc_Create();check(d,"Descriptor allocation failed");r.descriptors.push_back(d);
                    check(api.OH_NNTensorDesc_SetDataType(d,type)==OH_NN_SUCCESS,"Descriptor type failed");check(api.OH_NNTensorDesc_SetShape(d,shape.data(),shape.size())==OH_NN_SUCCESS,"Descriptor shape failed");check(api.OH_NNTensorDesc_SetFormat(d,OH_NN_FORMAT_ND)==OH_NN_SUCCESS,"Descriptor format failed");check(api.OH_NNModel_AddTensorToModel(r.model,d)==OH_NN_SUCCESS,"AddTensor failed");}
                uint32_t input=0,output=1;OH_NN_UInt32Array none{nullptr,0},ins{&input,1},outs{&output,1};
                check(api.OH_NNModel_AddOperation(r.model,OH_NN_OPS_SIN,&none,&ins,&outs)==OH_NN_SUCCESS,"AddSin failed");check(api.OH_NNModel_SpecifyInputsAndOutputs(r.model,&ins,&outs)==OH_NN_SUCCESS,"Model IO failed");check(api.OH_NNModel_Finish(r.model)==OH_NN_SUCCESS,"Model finish failed");
                const bool* supported=nullptr;uint32_t num=0;const auto query=api.OH_NNModel_GetAvailableOperations(r.model,selected,&supported,&num);
                row["supportQueryCode"]=int(query);row["supportListLength"]=num;row["supportAdvertised"]=query==OH_NN_SUCCESS&&num==1&&supported&&supported[0];
                r.compilation=api.OH_NNCompilation_Construct(r.model);check(r.compilation,"Compilation allocation failed");
                row["setDeviceCode"]=int(api.OH_NNCompilation_SetDevice(r.compilation,selected));check(row["setDeviceCode"]==0,"Set device failed");HiAI_ExecuteDevice npu=HIAI_EXECUTE_DEVICE_NPU;
                row["npuOnlyOptionCode"]=int(api.HMS_HiAIOptions_SetModelDeviceOrder(r.compilation,&npu,1));row["disableFallbackCode"]=int(api.HMS_HiAIOptions_SetFallbackMode(r.compilation,HIAI_FALLBACK_DISABLED));
                check(row["npuOnlyOptionCode"]==0&&row["disableFallbackCode"]==0,"NPU-only/no-fallback options rejected");
                const auto begin=std::chrono::steady_clock::now();const auto built=api.OH_NNCompilation_Build(r.compilation);row["buildCode"]=int(built);row["compileSeconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();row["onlineNpuBuildSucceeded"]=built==OH_NN_SUCCESS;
            }catch(const std::exception& error){row["error"]=error.what();}
            report["cases"].push_back(row);std::cout<<"LEVO_NPU_SUPPORT_CASE "<<row.dump()<<std::endl;
        }
        report["probeCompleted"]=true;
    }catch(const std::exception& error){report["error"]=error.what();}
    std::cout<<"LEVO_NPU_SUPPORT_REPORT "<<report.dump()<<std::endl;return report["probeCompleted"].get<bool>()?0:1;
}
