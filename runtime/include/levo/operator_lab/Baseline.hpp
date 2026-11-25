#pragma once
// Independent LLM executor: no DiT/VAE, sampling, model mutation or kernel hooks.
#include "../RuntimeSupport.hpp"
#include "../experimental/HiddenStateTap.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"
#include <iomanip>
#include <mutex>
#include <sstream>
#include "MlpCapture.hpp"

namespace LlmOperatorLab {
#ifndef LLM_OPERATOR_MNN_SHA256
#define LLM_OPERATOR_MNN_SHA256 "not-recorded"
#endif
#ifndef LLM_OPERATOR_VARIANT_ID
#define LLM_OPERATOR_VARIANT_ID "baseline"
#endif
extern "C" void LlmOperatorCandidateSetup() __attribute__((weak));
extern "C" uint64_t LlmOperatorCandidateCalls() __attribute__((weak));
inline std::atomic<bool> busy{false},cancelled{false};
inline std::mutex stateMutex;
inline Json state={{"busy",false},{"status","idle"}};
inline Json poll(){std::lock_guard<std::mutex> lock(stateMutex);auto value=state;value["busy"]=busy.load();return value;}
inline void patch(const Json& value){std::lock_guard<std::mutex> lock(stateMutex);for(auto i=value.begin();i!=value.end();++i)state[i.key()]=i.value();}
inline std::string caseName(int model){require(model==0||model==1,"Unknown baseline model");return model==0?"small":"large";}
inline void safeRelative(const std::string& value){
    require(!value.empty()&&value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-")==std::string::npos,"Invalid case file name");
    std::filesystem::path path(value);require(!path.is_absolute(),"Case path must be relative");
    for(const auto& part:path)require(part!="..","Case path traversal rejected");
}
inline Json inspect(const std::string& root,int model){
    Json result={{"ready",false},{"case",caseName(model)}};
    try{
        const auto dir=root+"/cases/"+caseName(model);auto manifest=readJson(dir+"/manifest.json");
        require(manifest["schemaVersion"]==1,"Unknown case schema");
        require(manifest.value("mnnArchiveSha256",std::string())==LLM_OPERATOR_MNN_SHA256,"Case and compiled MNN fingerprints differ");
        const auto profile=manifest.at("profile");require(profile["hiddenWidth"]==(model?2560:1024)&&profile["layers"]==(model?36:28),"Wrong trained-declared music LM architecture");
        require(profile["classes"]==16385&&profile["audioInputBase"]==(model?1:151830),"Wrong music token ABI");
        require(manifest["fixture"]["positions"]==513,"Baseline prefix must have513 positions");
        for(const auto& file:manifest.at("files")){
            const std::string name=file.at("relative");safeRelative(name);
            require(std::filesystem::is_regular_file(dir+"/"+name)&&std::filesystem::file_size(dir+"/"+name)==file.at("bytes").get<uint64_t>(),"Missing/wrong-size case asset: "+name);
        }
        result["ready"]=true;result["manifest"]=manifest;
        result["integrityScope"]="Host deployment verifies SHA256; App verifies architecture and file sizes";
    }catch(const std::exception& error){result["error"]=error.what();}
    return result;
}
struct Profiler {
    std::string phase;size_t ordinal=0;Json nodes=Json::object();
    MlpCapture capture;
    explicit Profiler(std::string captureRoot={}):capture(std::move(captureRoot)){}
    struct Pending {std::string key;Clock::time_point start;};std::vector<Pending> pending;
    static Json shapes(const std::vector<MNN::Tensor*>& tensors){Json result=Json::array();for(auto t:tensors)result.push_back(t->shape());return result;}
    static uint64_t logicalBytes(const std::vector<MNN::Tensor*>& tensors){uint64_t value=0;for(auto t:tensors)value+=uint64_t(t->elementSize())*((t->getType().bits+7)/8);return value;}
    MNN::TensorCallBackWithInfo before(){return [this](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){
        const std::string key=phase+"/"+std::to_string(ordinal++)+"/"+info->name();
        if(!nodes.contains(key))nodes[key]={{"id",key},{"phase",phase},{"name",info->name()},{"type",info->type()},{"calls",0},{"seconds",0.},{"maxSeconds",0.},
            {"inputShapes",shapes(tensors)},{"logicalInputBytes",logicalBytes(tensors)},{"mnnReportedFlops",info->flops()},{"backendObservations",Json::object()}};
        capture.before(phase,info->name(),tensors);
        pending.push_back({key,Clock::now()});return true;
    };}
    MNN::TensorCallBackWithInfo after(){return [this](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){
        const auto end=Clock::now();require(!pending.empty(),"Unmatched operator callback");auto p=pending.back();pending.pop_back();auto& row=nodes[p.key];
        const double elapsed=std::chrono::duration<double>(end-p.start).count();row["calls"]=row["calls"].get<int>()+1;row["seconds"]=row["seconds"].get<double>()+elapsed;row["maxSeconds"]=std::max(row["maxSeconds"].get<double>(),elapsed);
        row["outputShapes"]=shapes(tensors);row["logicalOutputBytes"]=logicalBytes(tensors);
        for(auto tensor:tensors){auto bn=MNN::TensorUtils::getDescribeOrigin(tensor)->getBackend();if(!bn)continue;const int type=int(bn->type());if(type!=0&&type!=13)continue;
            const auto core=static_cast<MNN::CPUBackend*>(bn)->functions();const std::string tag="backend"+std::to_string(type)+"/coreBytes"+std::to_string(core->bytes);
            auto& counts=row["backendObservations"];counts[tag]=counts.value(tag,0)+1;
        }
        capture.after(phase,info->name(),tensors);return true;
    };}
    Json report(){require(pending.empty(),"Incomplete operator callbacks");Json rows=Json::array();for(const auto& row:nodes)rows.push_back(row);
        std::sort(rows.begin(),rows.end(),[](const Json& a,const Json& b){return a["seconds"].get<double>()>b["seconds"].get<double>();});
        return {{"nodes",rows},{"capture",capture.report()},{"scope","Instrumented CPU callback wall time, separate from normal timing. IO bytes are logical tensor estimates; exclude implicit weights and internal KV reads; not measured DRAM/SRAM."}};
    }
};
inline void guard(const Json& options){
    require(!cancelled.load(),"Cancelled by user");require(resident()<options.value("maxRssMiB",6144)*1024L,"Diagnostic process RSS limit reached");
    require(available()>options.value("minAvailableMiB",650)*1024L,"System available memory below diagnostic gate");
}
inline void saveFloats(const std::string& path,const std::vector<float>& values){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*4);require(out.good(),"Cannot save private regression fixture");}
inline Json run(const std::string& root,int model,const Json& options){
    const int steps=options.value("decodeSteps",64),rounds=options.value("rounds",3),threads=options.value("threads",4);
    const std::string mode=options.value("mode",std::string("baseline")),precision=options.value("precision",std::string("high"));
    const std::string memory=options.value("memory",std::string("low"));
    require(memory=="low"||memory=="normal","Unknown operator-lab memory mode");
    require(memory!="normal"||(model==0&&precision=="high"&&available()>3400*1024L),
            "Float-weight reference requires the small case, CPU High and 3400 MiB available");
    require(mode=="baseline"||mode=="operators","Unknown executor mode");const bool instrumented=mode=="operators";
    require(steps>0&&steps<=3000&&rounds>0&&rounds<=5&&threads>=1&&threads<=8,"Baseline dimensions outside experiment bounds");
    require(!instrumented||(steps<=4&&rounds==1),"Operator diagnostics use1round/at most4decode, excluded from baseline performance");
    require(!options.value("captureMlp",false)||instrumented,"MLP capture is diagnostic-only");
    require(precision=="high"||precision=="low","Unknown CPU precision");
    auto checked=inspect(root,model);require(checked.value("ready",false),checked.value("error",std::string("Case not deployed")));
    const auto manifest=checked["manifest"];const int width=model?2560:1024;const bool hidden=model==1;
    const std::string dir=root+"/cases/"+caseName(model),id=std::to_string(std::chrono::system_clock::now().time_since_epoch().count()),output=root+"/runs/"+id;
    require(!std::filesystem::exists(output),"Preserve previous experiment");std::filesystem::create_directories(output);
    Json report={{"schemaVersion",1},{"runId",id},{"status","running"},{"system","B0_MNN_pinned"},{"mode",mode},{"memoryMode",memory},{"modelId",manifest["modelId"]},{"profile",manifest["profile"]},
        {"fixture",manifest["fixture"]},{"runTag",options.value("runTag",std::string())},{"precision",precision},{"threads",threads},{"instrumented",instrumented},
        {"nativeBackendSource","Existing pinned MNN archive; not claimed unmodified upstream"},{"mnnArchiveSha256",LLM_OPERATOR_MNN_SHA256},{"variantId",LLM_OPERATOR_VARIANT_ID},{"scope","LLM-only real music model, fixed teacher-code history. No DiT,VAE,sampling or generated audio; not whole-music RTF."},
        {"rounds",Json::array()},{"caseSignature",manifest.value("caseSignature",std::string())},{"requestedDecodeSteps",steps}};
    patch({{"busy",true},{"status","running"},{"runId",id},{"runTag",report["runTag"]},{"mode",mode},{"model",model},{"phase","load"},{"round",0},{"decodeDone",0},{"decodeSteps",steps}});
    Trace trace;const auto totalTick=Clock::now();
    try{
        guard(options);auto prompt=read(dir+"/fixtures/prefix.f32",size_t(513)*width);auto codes=readJson(dir+"/fixtures/codes.json").get<std::vector<int>>();
        require(codes.size()>=size_t(steps),"Insufficient real teacher history for requested decode length");
        for(int i=0;i<steps;++i)require(codes[i]>=0&&codes[i]<16384,"Invalid fixture music code");
        const int dynamic=manifest["profile"]["dynamicOption"];
        Json config={{"base_dir",dir+"/lm/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},{"llm_config","llm_config.json"},{"tokenizer_file","tokenizer.mtok"},{"embedding_file","embeddings_bf16.bin"},
            {"backend_type","cpu"},{"precision",precision},{"memory",memory},{"power","high"},{"thread_num",threads},{"reuse_kv",false},{"attention_mode",8},{"dynamic_option",dynamic},
            {"hidden_states",hidden},{"all_logits",false},{"cosyvoice_kleidiai",false},{"enable_debug",instrumented},{"tmp_path",output+"/"}};
        {std::ofstream out(output+"/config.json");out<<config.dump(2);}
        if(LlmOperatorCandidateSetup)LlmOperatorCandidateSetup();
        auto tick=Clock::now();std::unique_ptr<Llm,decltype(&Llm::destroy)> llm(Llm::createLLM(output+"/config.json"),&Llm::destroy);require(llm&&llm->load(),"Pinned music LM failed to load");report["loadMs"]=seconds(tick)*1000;
        const auto candidateBefore=LlmOperatorCandidateCalls?LlmOperatorCandidateCalls():0;
        Profiler profiler(options.value("captureMlp",false)?output+"/captures":std::string());if(instrumented)llm->setDebugCallback(profiler.before(),profiler.after());
        std::vector<float> referenceLogits,referenceHidden;Json checkpointSteps=Json::array();
        for(int round=0;round<rounds;++round){
            llm->reset();llm->generate_init();guard(options);patch({{"round",round+1},{"phase","prefill"},{"decodeDone",0}});
            auto input=MNN::Express::_Const(prompt.data(),{513,1,width},MNN::Express::NCHW,halide_type_of<float>());require(input->readMap<float>(),"Cannot prepare prefix");
            profiler.phase="prefill";profiler.ordinal=0;tick=Clock::now();auto logits=llm->forward(input);require(logits.get()&&logits->getInfo()->size==16385,"Unexpected baseline vocabulary");const float* p=logits->readMap<float>();require(p,"Prefill synchronization failed");const double prefill=seconds(tick)*1000;
            std::vector<float> capturedLogits,capturedHidden,hiddenRow(width);double maxLogitDiff=0,maxHiddenDiff=0,decode=0;Json tokenMs=Json::array();
            auto capture=[&](int step){for(int i=0;i<16385;++i)require(std::isfinite(p[i]),"Nonfinite baseline logits");
                const bool keep=steps<=256||step==0||step==1||step%250==0||step==steps;if(!keep)return;
                capturedLogits.insert(capturedLogits.end(),p,p+16385);if(round==0)checkpointSteps.push_back(step);
            };capture(0);
            patch({{"phase","decode"},{"prefillMs",prefill}});
            for(int step=0;step<steps;++step){
                guard(options);profiler.phase="decode";profiler.ordinal=0;tick=Clock::now();logits=llm->forward({(model?1:151830)+codes[step]},false);
                require(logits.get()&&logits->getInfo()->size==16385,"Decode vocabulary changed");p=logits->readMap<float>();require(p,"Decode synchronization failed");const double elapsed=seconds(tick)*1000;decode+=elapsed;tokenMs.push_back(elapsed);
                capture(step+1);if(hidden){LevoHiddenResearch::copySingleStepInto(*llm,hiddenRow.data(),width,LevoHiddenResearch::SingleStepLayout::Channels1x1);
                    if(steps<=256||step==0||(step+1)%250==0||step+1==steps)capturedHidden.insert(capturedHidden.end(),hiddenRow.begin(),hiddenRow.end());}
                if(step%16==15||step+1==steps)patch({{"decodeDone",step+1},{"decodeMsPerToken",decode/(step+1)},{"rssPeakMiB",trace.maxRss.load()/1024.}});
            }
            if(round==0){referenceLogits=capturedLogits;referenceHidden=capturedHidden;saveFloats(output+"/reference-logits.f32",referenceLogits);if(hidden)saveFloats(output+"/reference-hidden.f32",referenceHidden);}
            else{maxLogitDiff=relative(capturedLogits,referenceLogits);if(hidden)maxHiddenDiff=relative(capturedHidden,referenceHidden);}
            report["rounds"].push_back({{"kind",round?"warm":"first"},{"prefillPositions",513},{"prefillMs",prefill},{"decodeForwards",steps},{"decodeMs",decode},{"decodeMsPerToken",decode/steps},{"decodeTokensPerSecond",1000*steps/decode},{"kvPositions",llm->getCurrentHistory()},{"logitL2VsFirst",maxLogitDiff},{"hiddenL2VsFirst",maxHiddenDiff},{"decodeTokenMs",tokenMs}});
            auto summary=report["rounds"].back();summary.erase("decodeTokenMs");patch({{"roundResult",summary}});
        }
        report["candidateCalls"]=LlmOperatorCandidateCalls?LlmOperatorCandidateCalls()-candidateBefore:0;
        require(!options.value("requireCandidateCalls",false)||report["candidateCalls"].get<uint64_t>()>0,"Candidate was not observed during inference; do not benchmark silent baseline fallback");
        report["checkpointSteps"]=checkpointSteps;report["operators"]=instrumented?profiler.report():Json();report["status"]="completed";report["allOutputsFinite"]=true;
    }catch(const std::exception& error){report["status"]=cancelled.load()?"cancelled":"error";report["error"]=error.what();}
    trace.sample();report["elapsedSeconds"]=seconds(totalTick);report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["minimumAvailableMiB"]=trace.minAvailable.load()/1024.;
    report["memoryScope"]="20ms sampled whole-process RSS, not dedicated VRAM or SRAM";report["timingScope"]="Forward plus output synchronization, excludes finite checks, guard polling, snapshots and hidden copies. Operator mode is instrumented and excluded from baseline timing.";
    {std::ofstream out(output+"/report.json");out<<report.dump(2);require(out.good(),"Cannot save run report");}
    const std::string pending=root+"/llm-operator-report.json.pending";{std::ofstream out(pending);out<<report.dump(2);require(out.good(),"Cannot save unified report");}std::filesystem::rename(pending,root+"/llm-operator-report.json");
    Json summary=report;summary.erase("operators");summary.erase("rounds");if(!report["rounds"].empty()){summary["roundResult"]=report["rounds"].back();summary["roundResult"].erase("decodeTokenMs");}
    patch(summary);patch({{"busy",false},{"phase","done"},{"reportPath",root+"/llm-operator-report.json"}});return report;
}
}
