#include "MusicRuntime.hpp"
#include "ChainComparison.hpp"
#include <hidebug/hidebug.h>

static std::atomic<int> comparisonPhase{0};
static void comparisonStage(const char* text){std::string stage=text;
    if(stage.find("vae-start")!=std::string::npos||stage.find("wav-written")!=std::string::npos)comparisonPhase=3;
    else if(stage.rfind("chunk-",0)==0)comparisonPhase=2;
    else if(stage.find("prefill")!=std::string::npos||stage.find("decode")!=std::string::npos||stage.find("-lm")!=std::string::npos||stage.find("music-codes")!=std::string::npos)comparisonPhase=1;
    else comparisonPhase=0;
}
struct ComparisonMemory {
    std::atomic<bool> done{false};std::thread sampler;uint64_t rss[4]{},pss[4]{},graphics[4]{};
    uint64_t peakRss=0,peakPss=0,peakGraphics=0;int pssSamples=0,graphicsSamples=0,graphicsError=0;double querySeconds=0;
    ComparisonMemory(){comparisonPhase=0;levoStageHook=comparisonStage;sampler=std::thread([this]{int tick=0;
        while(!done){const int phase=comparisonPhase.load();uint64_t value=resident();rss[phase]=std::max(rss[phase],value);peakRss=std::max(peakRss,value);
            if(tick%25==0){auto begin=Clock::now();HiDebug_NativeMemInfo info{};OH_HiDebug_GetAppNativeMemInfoWithCache(&info,true);
                if(info.pss){++pssSamples;pss[phase]=std::max(pss[phase],uint64_t(info.pss));peakPss=std::max(peakPss,uint64_t(info.pss));}
                if(tick%50==0){uint32_t gpu=0;auto error=OH_HiDebug_GetGraphicsMemory(&gpu);graphicsError=int(error);
                    if(error==HIDEBUG_SUCCESS){++graphicsSamples;graphics[phase]=std::max(graphics[phase],uint64_t(gpu));peakGraphics=std::max(peakGraphics,uint64_t(gpu));}}
                querySeconds+=seconds(begin);
            }++tick;std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }});
    }
    Json finish(){done=true;if(sampler.joinable())sampler.join();levoStageHook=nullptr;Json stages=Json::object();
        const char* names[]={"initializationAndCondition","llm","dit","vae"};for(int k=0;k<4;++k)stages[names[k]]={{"peakRssKiB",rss[k]},{"sampledPeakPssKiB",pss[k]?Json(pss[k]):Json(nullptr)},{"sampledPeakGraphicsKiB",graphicsSamples?Json(graphics[k]):Json(nullptr)}};
        return {{"peakRssKiB",peakRss},{"sampledPeakPssKiB",pssSamples?Json(peakPss):Json(nullptr)},
            {"sampledPeakGraphicsKiB",graphicsSamples?Json(peakGraphics):Json(nullptr)},{"graphicsApiLastError",graphicsError},
            {"pssSamples",pssSamples},{"graphicsSamples",graphicsSamples},{"rssIntervalMs",20},{"pssIntervalMs",500},{"graphicsIntervalMs",1000},
            {"memoryQuerySeconds",querySeconds},{"perStage",stages},{"inferenceBackend","CPU"},
            {"interpretation","Shared physical memory; graphics includes App UI and is not model-only dedicated VRAM; sampled peaks may miss shorter transients"}};
    }
    ~ComparisonMemory(){if(sampler.joinable()){done=true;sampler.join();levoStageHook=nullptr;}}
};

static Json comparisonStudent(const std::string& root){
    PublicPhoneTrace trace;trace.sink=root+"/progress.jsonl";std::ofstream(trace.sink).close();levoPublicCancelled=false;
    Json report={{"success",false},{"diagnosticOnly",true},{"musicTrained",false},{"variant","small"},{"ditSteps",2}};
    try{
        trace.gate(2600);auto metadata=readJson(root+"/student/diagnostic-student.json");require(metadata["music_trained"]==false,"Expected diagnostic student");
        auto prefix=readJson(root+"/student/prefix.json").get<std::vector<int>>();require(prefix.size()==514,"Expected student 514-position prefix");
        auto config=readJson(root+"/student/config.json");config["base_dir"]=root+"/student/";config["thread_num"]=4;config["precision"]="low";
        config["memory"]="low";config["power"]="high";config["reuse_kv"]=false;config["attention_mode"]=8;config["cosyvoice_kleidiai"]=false;
        writePublicJson(root+"/student/comparison-config.json",config);auto tick=Clock::now();
        PublicLlm model(Llm::createLLM(root+"/student/comparison-config.json"),&Llm::destroy);require(model&&model->load(),"Student load failed");
        const double load=seconds(tick);model->reset();model->generate_init();trace.mark("student-lm-loaded");std::vector<int> codes;double prefill=0,decode=0;
        const int offset=metadata["music_offset"].get<int>();std::mt19937 random(2307);
        for(int i=0;i<750;++i){trace.gate(1200);tick=Clock::now();auto logits=publicLogits(*model,i?std::vector<int>{offset+codes.back()}:prefix,i==0);
            bool recent[16384]={false};for(size_t k=codes.size()>50?codes.size()-50:0;k<codes.size();++k)recent[codes[k]]=true;
            for(int k=0;k<16384;++k)if(recent[k])logits[k]/=1.1f;
            std::vector<int> indices(16384);std::iota(indices.begin(),indices.end(),0);
            std::partial_sort(indices.begin(),indices.begin()+50,indices.end(),[&](int a,int b){return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);});
            std::vector<double> weights(50);for(int k=0;k<50;++k)weights[k]=std::exp((double(logits[indices[k]])-logits[indices[0]])/.9);
            std::discrete_distribution<int> distribution(weights.begin(),weights.end());codes.push_back(indices[distribution(random)]);
            double duration=seconds(tick);if(i==0)prefill=duration;else decode+=duration;
            if(!i||codes.size()%25==0)trace.mark("real-phone-music-codes-"+std::to_string(codes.size()));
        }
        report["lm"]={{"loadSeconds",load},{"prefillSeconds",prefill},{"decodeSeconds",decode},{"decodeTokensPerSecond",749/decode},{"prefixPositions",514},
            {"finalKvPositions",model->getCurrentHistory()},{"eosSuppressedForFixedLengthDiagnostic",true},{"precision","low"},{"threads",4}};
        model.reset();report["lmStageSeconds"]=seconds(trace.start);trace.mark("student-lm-destroyed-before-codec");
        Json source={{"source","untrained 0.6B diagnostic student"},{"diagnosticOnly",true},{"lmDevice","Harmony phone CPU"},{"codes",codes},{"lmGuidance",1},
            {"progressPath",trace.sink}};writePublicJson(root+"/comparison-codes.json",source);
        std::string codePath=root+"/comparison-codes.json",steps="2",output="/dev/null",name="small";char* args[]={name.data(),const_cast<char*>(root.c_str()),codePath.data(),steps.data(),output.data()};Json codec;
        const int code=LevoPublicCodecMain(5,args,&codec);report["codec"]=codec;require(code==0,codec.value("error",std::string("Small codec failed")));
        report["validPcmSeconds"]=30;report["durationSeconds"]=30;report["pipelineSecondsThroughPcm"]=seconds(trace.start);report["pipelineRtfThroughPcm"]=seconds(trace.start)/30;
        report["success"]=true;report["audioFileGenerated"]=false;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();return report;
}

std::string LevoRunComparableChain(const std::string& root,const std::string& variant){
    ComparisonMemory memory;Json result;
    try{
        require(variant=="large-2"||variant=="small-2","Invalid comparison variant");
        if(variant=="small-2")result=comparisonStudent(root+"/small");
        else{auto input=readJson(root+"/input.json");result=musicGenerate(root+"/public",input["lyrics"],input["style"],750,2);
            result["variant"]="large";result["lmStageSeconds"]=result.contains("lm")?result["lm"]["totalLmSeconds"]:Json(nullptr);
            result["validPcmSeconds"]=result.value("durationSeconds",0.0);result["pipelineRtfThroughPcm"]=result.value("wholePhoneRtf",0.0);
        }
        result["memory"]=memory.finish();
        if(result.contains("sampledPeakRssKiB"))result["memory"]["peakRssKiB"]=std::max(result["memory"]["peakRssKiB"].get<uint64_t>(),result["sampledPeakRssKiB"].get<uint64_t>());
        if(result.value("success",false)){
            const double duration=result["durationSeconds"];double dit=0,vae=0;for(const auto& chunk:result["codec"]["chunks"]){dit+=chunk["ditSeconds"].get<double>();vae+=chunk["vaeSeconds"].get<double>();}
            const double llm=result["lmStageSeconds"],total=variant=="small-2"?result["pipelineSecondsThroughPcm"].get<double>():result["totalSeconds"].get<double>();
            result["stageRtf"]={{"llm",llm/duration},{"dit",dit/duration},{"vae",vae/duration},{"other",(total-llm-dit-vae)/duration},{"total",total/duration}};
            result["stageSeconds"]={{"llm",llm},{"dit",dit},{"vae",vae},{"other",total-llm-dit-vae},{"total",total}};
            result["stageSeconds"]["prefill"]=result["lm"]["prefillSeconds"];result["stageSeconds"]["decode"]=result["lm"]["decodeSeconds"];
            result["stageRtf"]["prefill"]=result["lm"]["prefillSeconds"].get<double>()/duration;
            result["stageRtf"]["decode"]=result["lm"]["decodeSeconds"].get<double>()/duration;
        }
    }catch(const std::exception& e){result={{"success",false},{"error",e.what()},{"memory",memory.finish()}};}
    result["lifetimePolicy"]="Serial LLM complete and destroy, then one DiT and one VAE session reused for all chunks";
    result["quantization"]={{"lmBodyWeightBits",4},{"lmHeadWeightBits",8},{"inputEmbeddingBits",16},{"ditWeightBits",8},{"vaeWeightBits",8},{"lmGuidance",1},{"ditGuidance",1.5},
        {"lmPrecision",variant=="small-2"?"Low":"High"},{"ditPrecision","High"},{"vaePrecision","Low"},{"lmThreads",4},{"codecThreads",6}};
    result["comparisonScope"]="Same App, 30-second equivalent, 2-step CFG DiT and same pretrained VAE; model sizes, condition contracts and training status differ";
    return result.dump();
}
