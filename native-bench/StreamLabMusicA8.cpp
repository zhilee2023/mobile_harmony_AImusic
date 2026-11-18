// Current trained-declared28L music LM, cropped W8 head; no language checkpoint.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include <MNN_generated.h>
using namespace LevoStreamLab;
namespace {
std::atomic<uint64_t> symmetric{0}, asymmetric{0}, scalarQuant{0};
decltype(MNN::CoreFunctions::MNNDynamicQuant) originalSym=nullptr;
decltype(MNN::CoreFunctions::MNNAsyQuantFunc) originalAsym=nullptr;
decltype(MNN::CoreInt8Functions::MNNFloat2Int8) originalFloat=nullptr;
void sym(const float* src,int8_t* dst,const float* scale,size_t depth,size_t count,int pack,const float* bias){++symmetric;originalSym(src,dst,scale,depth,count,pack,bias);}
void asym(int8_t* dst,const float* src,float* scale,float* bias,const size_t* info){++asymmetric;originalAsym(dst,src,scale,bias,info);}
void quant(const float* src,int8_t* dst,size_t size,const float* scale,ssize_t lo,ssize_t hi,const float* zero,ssize_t vec){++scalarQuant;originalFloat(src,dst,size,scale,lo,hi,zero,vec);}
struct Audit {
    MNN::CoreFunctions* core=MNN::MNNGetCoreFunctions();MNN::CoreInt8Functions* integer=MNN::MNNGetInt8CoreFunctions();
    explicit Audit(bool enabled){if(!enabled){core=nullptr;integer=nullptr;return;}originalSym=core->MNNDynamicQuant;originalAsym=core->MNNAsyQuantFunc;originalFloat=integer->MNNFloat2Int8;
        require(originalSym&&originalAsym&&originalFloat,"Pinned dynamic A8 hooks missing");core->MNNDynamicQuant=sym;core->MNNAsyQuantFunc=asym;integer->MNNFloat2Int8=quant;}
    ~Audit(){if(core){core->MNNDynamicQuant=originalSym;core->MNNAsyQuantFunc=originalAsym;integer->MNNFloat2Int8=originalFloat;}}
};
}
int main(int argc,char** argv){
    if(argc==2&&std::string(argv[1])=="--sampling-contract"){
        Json tests=Json::array();
        try{
            for(const auto& v:std::vector<MusicVocabulary>{{168214,151830,151765,151830},{16449,65,0,151830},{16385,1,0,151830},{16385,1,0,1}}){
                std::vector<float> logits(v.classes,1000000.f);for(int i=0;i<16384;++i)logits[v.audioBase+i]=-1000;
                logits[v.eos]=1000;std::mt19937 random(2307);require(sampleMusicCode(logits.data(),v,false,random)==-1,"EOS was treated as an audio code");
                logits[v.audioBase+16383]=1000;require(sampleMusicCode(logits.data(),v,true,random)==16383,"Fixed-duration suppression changed the last music code");
                tests.push_back({{"classes",v.classes},{"audioOutputBase",v.audioBase},{"eosOutput",v.eos},{"eosStopsNormally",true},{"fixedDurationCanSuppressEos",true},{"lastAudioInputId",v.inputAudioBase+16383}});
            }
            std::cout<<Json({{"success",true},{"scope","Actual shared sampler, synthetic decisive logits; does not measure model music quality or RTF"},{"variants",tests}}).dump()<<std::endl;return 0;
        }catch(const std::exception& e){std::cout<<Json({{"success",false},{"error",e.what()}}).dump()<<std::endl;return 1;}
    }
    if(argc!=8)return 2;const std::string root=argv[1],crop=argv[2],output=argv[5],mode=argv[7];const int option=std::stoi(argv[6]);const bool audit=mode=="audit",floating=mode=="reference";
    Json report={{"success",false},{"scope","Actual28-layer music Qwen, real513-position lyric/style prefix; fixed speech-code history, not complete music generation"},
                 {"dynamicOption",option},{"headWeightBits",8},{"bodyWeightBits",4},{"attentionMode",8},{"kvConfigurationChanged",false},{"auditHooksEnabled",audit},
                 {"floatingExpandedWeightsReference",floating},{"referenceMeaning","Same already-W4/W8 parameters expanded to float, isolates extra activation/kernel error; not original prequantized checkpoint"}};Trace trace;
    try{
        require(option==0||option==2,"Only default/global and per-weight-block dynamic A8 options supported");require(!std::filesystem::exists(output),"Preserve old LM outputs");std::filesystem::create_directories(output);
        require(mode=="perf"||mode=="audit"||mode=="reference","Unknown music probe mode");
        const auto request=readJson(argv[3]);auto codes=readJson(argv[4]).get<std::vector<int>>();require(codes.size()>=128,"Need128 real sampled music codes");
        const auto vocab=musicVocabulary(true,crop);report["musicVocabulary"]={{"classes",vocab.classes},{"audioOutputBase",vocab.audioBase},{"eosOutput",vocab.eos},{"inputAudioBase",vocab.inputAudioBase}};
        std::ifstream file(crop+"/llm.mnn",std::ios::binary|std::ios::ate);require(bool(file),"Missing music graph");std::vector<char> bytes(size_t(file.tellg()));file.seekg(0);file.read(bytes.data(),bytes.size());
        auto net=MNN::GetNet(bytes.data());Json counts=Json::object();for(auto op:*net->oplists()){auto c=op->main_as_Convolution2D();if(c&&c->quanParameter()){auto q=c->quanParameter();auto key=std::to_string(q->aMaxOrBits());counts[key]=counts.value(key,0)+1;
            if(c->common()->outputCount()==vocab.classes)report["headMetadata"]={{"bitsField",q->aMaxOrBits()},{"readType",q->readType()},{"quantType",q->type()},{"channels",c->common()->outputCount()}};}}
        report["linearQuantBitsFields"]=counts;
        auto prompt=prefix(root,1024,true,request.at("lyrics"),request.at("npy"));
        Json cfg={{"base_dir",crop+"/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},{"llm_config","llm_config.json"},{"tokenizer_file","tokenizer.mtok"},{"embedding_file","embeddings_bf16.bin"},
            {"backend_type","cpu"},{"precision","high"},{"memory",floating?"normal":"low"},{"power","high"},{"thread_num",4},{"reuse_kv",false},{"attention_mode",8},{"dynamic_option",option},
            {"hidden_states",false},{"all_logits",false},{"cosyvoice_kleidiai",false},{"tmp_path",output+"/"}};
        {std::ofstream out(output+"/config.json");out<<cfg.dump(2);}auto tick=Clock::now();std::unique_ptr<Llm,decltype(&Llm::destroy)> model(Llm::createLLM(output+"/config.json"),&Llm::destroy);
        require(model&&model->load(),"Music LM load failed");report["loadSeconds"]=seconds(tick);Audit hooks(audit);report["rounds"]=Json::array();
        std::vector<float> first;const int rounds=(audit||floating)?1:2,steps=audit?2:128;
        for(int round=0;round<rounds;++round){model->reset();model->generate_init();double prefill=0,decode=0;Json greedy=Json::array();std::vector<float> logits;
            for(int step=0;step<steps;++step){check();auto begin=Clock::now();auto v=step?model->forward({151830+codes[step-1]},false):model->forward(MNN::Express::_Const(prompt.data(),{513,1,1024},MNN::Express::NCHW,halide_type_of<float>()));
                require(v.get()&&v->getInfo()->size==vocab.classes,"Music logits contract changed");const float* p=v->readMap<float>();require(p,"Logit sync failed");const auto elapsed=seconds(begin);if(step)decode+=elapsed;else prefill=elapsed;
                for(int k=0;k<vocab.classes;++k)require(std::isfinite(p[k]),"Nonfinite music logit");greedy.push_back(int(std::max_element(p,p+vocab.classes)-p));logits.insert(logits.end(),p,p+vocab.classes);
            }
            Json row={{"prefillPositions",513},{"prefillSeconds",prefill},{"decodeForwards",steps-1},{"decodeSeconds",decode},{"decodeForwardsPerSecond",(steps-1)/decode},{"greedyOutputIndices",greedy},{"kvPositions",model->getCurrentHistory()}};
            if(round==0){first=logits;std::ofstream out(output+"/logits.f32",std::ios::binary);out.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4);}else row["sameHistoryLogitL2VsFirst"]=relative(logits,first);
            report["rounds"].push_back(row);std::cout<<"MUSIC_A8_ROUND "<<Json({{"round",round},{"prefill",prefill},{"decodePerSecond",(steps-1)/decode}}).dump()<<std::endl;
        }
        report["activationQuantKernelAudit"]={{"symmetricInt8DestinationCalls",symmetric.load()},{"asymmetricInt8DestinationCalls",asymmetric.load()},{"floatToInt8DestinationCalls",scalarQuant.load()},
            {"meaning","Actual int8 temporary buffers inside linear kernels, not all graph activations stored A8; audit excluded from speed cases"}};report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["minimumAvailableMiB"]=trace.minAvailable.load()/1024.;
    if(std::filesystem::is_directory(output))std::ofstream(output+"/result.json")<<report.dump(2);std::cout<<"MUSIC_A8_DONE "<<report.value("success",false)<<std::endl;return report["success"]==true?0:1;
}
