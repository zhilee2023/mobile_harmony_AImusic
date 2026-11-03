// Optional CLI probe; neither App includes this implementation.
#include "../runtime/include/levo/RuntimeSupport.hpp"
#include "../runtime/include/levo/experimental/HiddenStateTap.hpp"
#include <numeric>

static void saveFloats(const std::string& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float));
    require(file.good(), "Cannot save numerical reference");
}

int main(int argc, char** argv) {
    if (argc != 7) { std::cerr << "studentChainRoot workDir logits|allocated|buffer threads frames output.json\n"; return 2; }
    if (std::filesystem::exists(argv[6])) { std::cerr<<"Preserve prior report\n"; return 2; }
    Trace trace;
    Json report={{"success",false},{"scope","Fresh CLI process, fixed-token student LLM only; no sampling, conditioner, DiT, VAE or music RTF"}};
    try {
        const std::string root=argv[1], work=argv[2], mode=argv[3];
        const int threads=std::stoi(argv[4]), frames=std::stoi(argv[5]);
        require(mode=="logits"||mode=="allocated"||mode=="buffer", "Unknown output collection mode");
        require((threads==2||threads==3||threads==4||threads==6)&&frames>=2&&frames<=256, "Bounded probe sizes");
        std::filesystem::create_directories(work);
        trace.sink=work+"/progress.jsonl";
        const auto metadata=readJson(root+"/student/diagnostic-student.json");
        const int width=metadata["hidden"], offset=metadata["music_offset"];
        const auto prefix=readJson(root+"/student/prefix.json").get<std::vector<int>>();
        const auto teacher=readJson(root+"/comparison-codes.json")["codes"].get<std::vector<int>>();
        require(prefix.size()==514&&teacher.size()>=size_t(frames-1)&&width==1024,"Wrong student fixture");
        for(int code:teacher) require(code>=0&&code<16384,"Illegal teacher music code");
        auto config=readJson(root+"/student/config.json");
        config["base_dir"]=root+"/student/"; config["thread_num"]=threads;
        config["precision"]="low"; config["memory"]="low"; config["power"]="high";
        config["reuse_kv"]=false; config["attention_mode"]=8; config["cosyvoice_kleidiai"]=false;
        config["all_logits"]=false; config["hidden_states"]=mode!="logits";
        const std::string configPath=work+"/config.json";
        {std::ofstream out(configPath);out<<config.dump(2);require(out.good(),"Cannot write isolated config");}
        trace.gate(2400); auto tick=Clock::now();
        std::unique_ptr<Llm,decltype(&Llm::destroy)> model(Llm::createLLM(configPath),&Llm::destroy);
        require(model&&model->load(),"Cannot load fixed student LM");
        report["loadSeconds"]=seconds(tick);
        report["mode"]=mode; report["requestedThreads"]=threads;
        report["hiddenWidth"]=width; report["prefixPositions"]=prefix.size(); report["conditionFrames"]=frames;
        report["recipe"]={{"bodyWeightBits",4},{"headWeightBits",8},{"activationBitsRequested",nullptr},{"kvBitsRequested",nullptr},{"precision","low"}};
        trace.mark("hidden-probe-loaded");
        Json rounds=Json::array();
        for(int round=0;round<2;++round) {
            model->reset(); model->generate_init();
            std::vector<std::vector<float>> perToken;
            if(mode=="allocated") perToken.reserve(frames);
            std::vector<float> hidden(mode=="logits"?0:frames*width), logitsStorage(frames*16385), oneLogits(16385);
            double prefill=0,decode=0,logitCopy=0,handoff=0;
            const auto roundStart=Clock::now();
            for(int i=0;i<frames;++i) {
                require(seconds(trace.start)<180,"Probe deadline exceeded");
                if(i%32==0)trace.gate(1200);
                const std::vector<int> ids=i?std::vector<int>{offset+teacher[i-1]}:prefix;
                tick=Clock::now(); auto logits=model->forward(ids,i==0);
                require(logits.get()&&logits->getInfo(),"Forward failed");
                const float* ptr=logits->readMap<float>();
                require(ptr&&logits->getInfo()->size==16385,"Expected last-position music logits only");
                const double forward=seconds(tick); if(i==0)prefill=forward;else decode+=forward;
                tick=Clock::now();std::copy_n(ptr,16385,oneLogits.data());logitCopy+=seconds(tick);
                if(mode!="logits") {
                    if(i==0&&round==0) {
                        const auto outputs=model->getOutputs();const int index=model->getOutputIndex("hidden_states");
                        require(index>=0&&index<int(outputs.size())&&outputs[index].get(),"No hidden output");
                        const auto info=outputs[index]->getInfo();require(info,"No hidden metadata");
                        report["firstHiddenShape"]=info->dim;report["firstHiddenElements"]=info->size;
                        report["firstHiddenType"]={{"code",info->type.code},{"bits",info->type.bits}};
                    }
                    tick=Clock::now();
                    if(mode=="allocated")perToken.push_back(LevoHiddenResearch::copySingleStep(*model,width,LevoHiddenResearch::SingleStepLayout::Channels1x1));
                    else LevoHiddenResearch::copySingleStepInto(*model,hidden.data()+i*width,width,LevoHiddenResearch::SingleStepLayout::Channels1x1);
                    handoff+=seconds(tick);
                }
                // Reference snapshot copy is outside the above stage counters,
                // identically present in all modes, and included in wall time.
                std::copy(oneLogits.begin(),oneLogits.end(),logitsStorage.data()+i*16385);
            }
            if(mode=="allocated") {
                tick=Clock::now();
                for(int i=0;i<frames;++i)std::copy(perToken[i].begin(),perToken[i].end(),hidden.data()+i*width);
                handoff+=seconds(tick);
            }
            const double wall=seconds(roundStart);
            peak(logitsStorage); if(!hidden.empty())peak(hidden);
            if(round==0) {
                saveFloats(work+"/logits.f32",logitsStorage);
                if(!hidden.empty())saveFloats(work+"/hidden.f32",hidden);
            }
            rounds.push_back({{"round",round+1},{"prefillSeconds",prefill},{"decodeSeconds",decode},
                {"decodeForwards",frames-1},{"decodeForwardsPerSecond",(frames-1)/decode},
                {"logitsCopySeconds",logitCopy},{"hiddenHandoffSeconds",handoff},{"roundWallSeconds",wall},
                {"hiddenStorageBytes",hidden.size()*sizeof(float)},{"extraPerTokenHiddenAllocations",mode=="allocated"?frames:0},
                {"kvPositions",model->getCurrentHistory()}});
            trace.mark("hidden-probe-round-"+std::to_string(round+1));
        }
        report["rounds"]=rounds;model.reset();trace.mark("hidden-probe-destroyed");report["success"]=true;
    } catch(const std::exception& error) { report["error"]=error.what(); }
    trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();
    report["totalSeconds"]=seconds(trace.start);report["stages"]=trace.stages;
    if(argc==7){std::ofstream out(argv[6]);out<<report.dump(2);}
    std::cout<<"LEVO_HIDDEN_BENCH_REPORT "<<report.dump()<<std::endl;
    return report["success"].get<bool>()?0:1;
}
