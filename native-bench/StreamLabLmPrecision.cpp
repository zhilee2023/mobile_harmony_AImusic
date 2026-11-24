// Same trained-declared music LM and teacher history; isolate CPU precision.
// Debug callbacks/kernel hooks run separately from performance measurements.
#include "levo/experimental/StreamLab.hpp"
#include "backend/arm82/Arm82Functions.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/TensorUtils.hpp"
#include <array>
#include <utility>
using namespace LevoStreamLab;
namespace {
using Kernel=decltype(MNN::MatmulRelatedFunctions::Int8GemmKernel);
constexpr size_t MaxHooks=64;
std::array<Kernel,MaxHooks> originals{};
std::array<std::atomic<uint64_t>,MaxHooks> calls{};
template<size_t I> void counted(int8_t* dst,const int8_t* src,const int8_t* weight,
 size_t depth,size_t stride,size_t outputs,const QuanPostTreatParameters* post,size_t count){
    ++calls[I];originals[I](dst,src,weight,depth,stride,outputs,post,count);
}
template<size_t... I> auto wrappers(std::index_sequence<I...>){return std::array<Kernel,sizeof...(I)>{&counted<I>...};}
struct Audit {
    std::vector<std::pair<Kernel*,Kernel>> restore;
    std::vector<std::string> labels;
    void bind(Kernel& field,const std::string& label){
        if(!field)return;
        require(labels.size()<MaxHooks,"Kernel hook capacity exceeded");
        static const auto trampoline=wrappers(std::make_index_sequence<MaxHooks>{});
        const auto index=labels.size();originals[index]=field;restore.emplace_back(&field,field);
        labels.push_back(label);field=trampoline[index];
    }
    void table(MNN::MatmulRelatedFunctions& t,const std::string& prefix){
#define HOOK(field) bind(t.field,prefix+"." #field)
        HOOK(Int8GemmKernel);HOOK(Int8GemmKernel_W4);
        HOOK(MNNGemmInt8AddBiasScale_Unit_FP16);HOOK(MNNGemmInt8AddBiasScale_w4_Unit_FP16);
        HOOK(MNNGemmInt8AddBiasScale_Unit_FP16_DecodeMax);HOOK(MNNGemmInt8AddBiasScale_w4_Unit_FP16_DecodeMax);
        HOOK(MNNGemmInt8AddBiasScale_Unit_FP32_DecodeMax);HOOK(MNNGemmInt8AddBiasScale_w4_Unit_FP32_DecodeMax);
#undef HOOK
    }
    explicit Audit(bool enabled){
        if(!enabled)return;
        auto cpu=MNN::MNNGetCoreFunctions();auto arm=MNN::Arm82Functions::get();
        require(cpu&&arm,"Missing initialized CPU/ARM82 function tables");
        table(cpu->int8MatmulRelatedFunctions,"cpu.int8");table(cpu->arm82MatmulRelatedFunctions,"cpu.arm82");
        if(arm!=cpu){table(arm->int8MatmulRelatedFunctions,"arm82.int8");table(arm->arm82MatmulRelatedFunctions,"arm82.arm82");}
    }
    ~Audit(){for(auto& item:restore)*item.first=item.second;}
    Json result()const{Json result=Json::object();for(size_t i=0;i<labels.size();++i)if(calls[i])result[labels[i]]=calls[i].load();return result;}
};
void writeFloats(const std::string& path,const std::vector<float>& values){
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
    require(out.good(),"Cannot save comparison tensor");
}
}
int main(int argc,char** argv){
    if(argc!=8){std::cerr<<"MODEL_ROOT PREFIX_F32 CODES_JSON OUTPUT_NEW_DIR high|low perf|audit DECODE_COUNT\n";return 2;}
    const std::string root=argv[1],output=argv[4],precision=argv[5],mode=argv[6];
    const bool audit=mode=="audit";const int steps=std::stoi(argv[7]);
    Json report={{"success",false},{"scope","Actual36-layer music LM; same513-position prefix and teacher-code history, consume-aligned hidden; not whole-chain RTF"},
        {"precision",precision},{"auditEnabled",audit},{"bodyWeightBits",4},{"headWeightBits",8},{"dynamicOption",0},{"attentionMode",8},{"threads",4}};
    Trace trace;bool outputCreated=false;
    try{
        require(precision=="high"||precision=="low","Unknown precision");require(mode=="perf"||audit,"Unknown mode");
        require(steps>0&&steps<=256&&(!audit||steps<=4),"Bounded probe only");
        require(!std::filesystem::exists(output),"Preserve existing probe results");std::filesystem::create_directories(output);outputCreated=true;
        auto prompt=read(argv[2],513*2560);auto codes=readJson(argv[3]).get<std::vector<int>>();
        require(codes.size()>=size_t(steps),"Insufficient teacher codes");
        for(int i=0;i<steps;++i)require(codes[i]>=0&&codes[i]<16384,"Invalid teacher music code");
        // Initialize the same pinned backend registries before installing diagnostic hooks.
        MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;
        MNN::ScheduleConfig schedule;schedule.type=MNN_FORWARD_CPU;schedule.numThread=4;schedule.backendConfig=&backend;
        auto initialized=MNN::Interpreter::createRuntime({schedule});
        if(!MNN::Arm82Functions::get())require(MNN::Arm82Functions::init(),"ARM82 initialization failed");
        Audit hooks(audit);
        Json config={{"base_dir",root+"/lm/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},{"llm_config","llm_config.json"},
            {"tokenizer_file","tokenizer.mtok"},{"embedding_file","embeddings_bf16.bin"},{"backend_type","cpu"},{"precision",precision},{"memory","low"},
            {"power","high"},{"thread_num",4},{"reuse_kv",false},{"attention_mode",8},{"dynamic_option",0},{"hidden_states",true},
            {"all_logits",false},{"cosyvoice_kleidiai",false},{"enable_debug",audit},{"tmp_path",output+"/"}};
        {std::ofstream out(output+"/config.json");out<<config.dump(2);}
        auto tick=Clock::now();std::unique_ptr<Llm,decltype(&Llm::destroy)> model(Llm::createLLM(output+"/config.json"),&Llm::destroy);
        require(model&&model->load(),"Music LM failed to load");report["loadMs"]=1000*seconds(tick);
        Json backends=Json::object();
        if(audit){
            auto callback=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* op){
                for(auto tensor:tensors){auto bn=MNN::TensorUtils::getDescribeOrigin(tensor)->getBackend();if(!bn)continue;
                    const auto type=int(bn->type());if(type!=0&&type!=13)continue;
                    auto cpu=static_cast<MNN::CPUBackend*>(bn);const auto key=op->type()+std::string("/backend=")+std::to_string(type)+"/coreBytes="+std::to_string(cpu->functions()->bytes);
                    backends[key]=backends.value(key,uint64_t(0))+1;
                }return true;
            };
            model->setDebugCallback([](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo*){return true;},std::move(callback));
        }
        model->reset();model->generate_init();auto input=MNN::Express::_Const(prompt.data(),{513,1,2560},MNN::Express::NCHW,halide_type_of<float>());
        require(input->readMap<float>(),"Cannot prepare prefix");
        std::vector<float> logits,hidden(size_t(steps)*2560);Json greedy=Json::array();double decode=0,lookup=0;
        tick=Clock::now();auto v=model->forward(input);require(v.get()&&v->getInfo()->size==16385,"Invalid prefill output");
        const float* p=v->readMap<float>();require(p,"Cannot synchronize prefill");report["prefillMs"]=1000*seconds(tick);
        auto capture=[&](){for(int k=0;k<16385;++k)require(std::isfinite(p[k]),"Nonfinite logits");
            greedy.push_back(int(std::max_element(p,p+16385)-p));logits.insert(logits.end(),p,p+16385);};capture();
        std::cout<<"PRECISION_PREFILL "<<precision<<" "<<report["prefillMs"]<<std::endl;
        for(int i=0;i<steps;++i){
            trace.gate(650);tick=Clock::now();v=model->forward({1+codes[i]},false);
            require(v.get()&&v->getInfo()->size==16385,"Invalid decode output");p=v->readMap<float>();require(p,"Cannot synchronize decode");decode+=seconds(tick);
            capture();tick=Clock::now();LevoHiddenResearch::copySingleStepInto(*model,hidden.data()+size_t(i)*2560,2560,LevoHiddenResearch::SingleStepLayout::Channels1x1);lookup+=seconds(tick);
            if(i%16==15)std::cout<<"PRECISION_DECODE "<<precision<<" "<<i+1<<" "<<decode*1000/(i+1)<<std::endl;
        }
        report["prefillPositions"]=513;report["decodeForwards"]=steps;report["decodeMs"]=decode*1000;report["decodeMsPerToken"]=decode*1000/steps;
        report["decodeTokensPerSecond"]=steps/decode;report["hiddenCopyMs"]=lookup*1000;report["kvPositions"]=model->getCurrentHistory();report["greedyIndices"]=greedy;
        report["backendTensorObservations"]=backends;report["quantKernelCalls"]=hooks.result();
        report["auditTimingExcludedFromComparison"]=audit;
        writeFloats(output+"/logits.f32",logits);writeFloats(output+"/hidden.f32",hidden);report["success"]=true;
    }catch(const std::exception& error){report["error"]=error.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["minimumAvailableMiB"]=trace.minAvailable.load()/1024.;
    if(outputCreated)std::ofstream(output+"/result.json")<<report.dump(2);
    std::cout<<"PRECISION_DONE "<<report.dump()<<std::endl;return report.value("success",false)?0:1;
}
