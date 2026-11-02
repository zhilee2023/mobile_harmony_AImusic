#include "MusicRuntime.hpp"
#include "ChainComparison.hpp"
#include <hidebug/hidebug.h>

static std::atomic<int> comparisonPhase{0};
static void comparisonStage(const char* text){std::string stage=text;
    if(stage.find("vae")!=std::string::npos||stage.find("wav-written")!=std::string::npos)comparisonPhase=3;
    else if(stage.rfind("chunk-",0)==0||stage.find("dit")!=std::string::npos||stage=="real-public-codes-start")comparisonPhase=2;
    else if(stage.find("prefill")!=std::string::npos||stage.find("decode")!=std::string::npos||stage.find("-lm")!=std::string::npos||stage.find("music-codes")!=std::string::npos)comparisonPhase=1;
    else comparisonPhase=0;
}
struct ComparisonMemory {
    std::atomic<bool> done{false};std::thread sampler,profiler;uint64_t rss[4]{},pss[4]{},graphics[4]{};
    uint64_t peakRss=0,peakPss=0,peakGraphics=0;int pssSamples=0,graphicsSamples=0,graphicsError=0,boundaryQueries=0;double querySeconds=0;
    ComparisonMemory(){comparisonPhase=0;levoStageHook=comparisonStage;sampler=std::thread([this]{
        while(!done){const int phase=comparisonPhase.load();uint64_t value=resident();rss[phase]=std::max(rss[phase],value);peakRss=std::max(peakRss,value);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }});
        profiler=std::thread([this]{int tick=0;while(!done){int phase=comparisonPhase.load();auto begin=Clock::now();HiDebug_NativeMemInfo info{};OH_HiDebug_GetAppNativeMemInfoWithCache(&info,true);
            const bool stable=phase==comparisonPhase.load();if(!stable)++boundaryQueries;
            if(info.pss){++pssSamples;peakPss=std::max(peakPss,uint64_t(info.pss));if(stable)pss[phase]=std::max(pss[phase],uint64_t(info.pss));}
            if(tick%2==0){phase=comparisonPhase.load();uint32_t gpu=0;auto error=OH_HiDebug_GetGraphicsMemory(&gpu);graphicsError=int(error);
                if(error==HIDEBUG_SUCCESS){++graphicsSamples;peakGraphics=std::max(peakGraphics,uint64_t(gpu));if(phase==comparisonPhase.load())graphics[phase]=std::max(graphics[phase],uint64_t(gpu));else ++boundaryQueries;}}
            querySeconds+=seconds(begin);++tick;for(int i=0;i<25&&!done;++i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }});
    }
    Json finish(){done=true;if(sampler.joinable())sampler.join();if(profiler.joinable())profiler.join();levoStageHook=nullptr;Json stages=Json::object();
        const char* names[]={"initializationAndCondition","llm","dit","vae"};for(int k=0;k<4;++k)stages[names[k]]={{"peakRssKiB",rss[k]},{"sampledPeakPssKiB",pss[k]?Json(pss[k]):Json(nullptr)},{"sampledPeakGraphicsKiB",graphicsSamples?Json(graphics[k]):Json(nullptr)}};
        return {{"peakRssKiB",peakRss},{"sampledPeakPssKiB",pssSamples?Json(peakPss):Json(nullptr)},
            {"sampledPeakGraphicsKiB",graphicsSamples?Json(peakGraphics):Json(nullptr)},{"graphicsApiLastError",graphicsError},
            {"pssSamples",pssSamples},{"graphicsSamples",graphicsSamples},{"rssIntervalMs",20},{"pssIntervalMs",500},{"graphicsIntervalMs",1000},
            {"memoryQuerySeconds",querySeconds},{"perStage",stages},{"inferenceBackend","CPU"},{"independentRssSampler",true},{"crossStageQueriesExcludedFromPerStage",boundaryQueries},
            {"interpretation","Shared physical memory; graphics includes App UI and is not model-only dedicated VRAM; sampled peaks may miss shorter transients"}};
    }
    ~ComparisonMemory(){done=true;if(sampler.joinable())sampler.join();if(profiler.joinable())profiler.join();levoStageHook=nullptr;}
};

std::string LevoRunCodec120Benchmark(const std::string& directory,int frames){
    const std::string root=directory+"/public";ComparisonMemory memory;PublicPhoneTrace trace;trace.sink=root+"/progress.jsonl";
    std::ofstream(trace.sink).close();levoPublicCancelled=false;
    const double duration=frames/25.;
    Json result={{"success",false},{"codecOnly",true},{"wholeChainCompleted",false},{"llmExecuted",false},{"durationSeconds",duration},{"ditSteps",2}};
    try{
        auto source=readJson(root+(frames==2250?"/codec-90-source.json":"/codec-120-source.json"));auto codes=source["codes"].get<std::vector<int>>();
        require(!codes.empty()&&codes.size()<=size_t(frames),"Invalid codec benchmark fixture");size_t original=codes.size();
        for(size_t i=0;codes.size()<size_t(frames);++i)codes.push_back(codes[i%original]);
        const bool repeated=original<size_t(frames);
        source["codes"]=codes;source["performanceFixtureOnly"]=repeated;source["progressPath"]=trace.sink;
        source["lmGuidance"]=1;source["lmDevice"]=repeated?"Earlier phone partial checkpoint; repeated fixture, no LLM in this run":"Earlier phone complete-code checkpoint; no repetition and no LLM in this recovery process";
        result["originalRealCodes"]=original;result["repeatedFixtureCodes"]=frames-original;
        writePublicJson(root+"/codec-120-fixture.json",source);trace.gate(2600);
        std::string file=root+"/codec-120-fixture.json",steps="2",output="/dev/null",name="codec-only";
        if(!repeated){std::filesystem::create_directories(root+"/songs");output=root+"/songs/recovered-"+std::to_string(frames/25)+"-"+std::to_string(std::chrono::system_clock::now().time_since_epoch().count())+".wav";result["wave"]=output;}
        char* args[]={name.data(),const_cast<char*>(root.c_str()),file.data(),steps.data(),output.data()};Json codec;
        int error=LevoPublicCodecMain(5,args,&codec);result["codec"]=codec;require(error==0,codec.value("error",std::string("Codec-only failed")));
        double dit=0,vae=0;for(const auto& row:codec["chunks"]){dit+=row["ditSeconds"].get<double>();vae+=row["vaeSeconds"].get<double>();}
        double total=seconds(trace.start);result["stageSeconds"]={{"llm",nullptr},{"dit",dit},{"vae",vae},{"other",total-dit-vae},{"totalCodecOnly",total}};
        result["stageRtf"]={{"llm",nullptr},{"dit",dit/duration},{"vae",vae/duration},{"other",(total-dit-vae)/duration},{"totalCodecOnly",total/duration}};
        result["codecOnlyRtf"]=total/duration;result["wholeChainRtf"]=nullptr;result["success"]=true;
    }catch(const std::exception& e){result["error"]=e.what();}
    result["memory"]=memory.finish();trace.sample();result["sampledPeakRssKiB"]=trace.maxRss.load();result["minimumSystemAvailableKiB"]=trace.minAvailable.load();
    result["quantization"]={{"ditWeightBits",8},{"vaeWeightBits",8},{"ditPrecision","High"},{"vaePrecision","Low"},{"codecThreads",6},{"ditGuidance",1.5}};
    result["scope"]="Codec-only continuation; originalRealCodes/repeatedFixtureCodes distinguish authentic complete sequence from a padded fixture. No single-process whole-chain RTF.";
    return result.dump();
}

static Json comparisonStudent(const std::string& root,int frames){
    PublicPhoneTrace trace;trace.sink=root+"/progress.jsonl";std::ofstream(trace.sink).close();levoPublicCancelled=false;
    Json report={{"success",false},{"diagnosticOnly",true},{"musicTrained",false},{"variant","small"},{"ditSteps",2}};
    try{
        trace.gate(2600);auto metadata=readJson(root+"/student/diagnostic-student.json");require(metadata["music_trained"]==false,"Expected diagnostic student");
        auto prefix=readJson(root+"/student/prefix.json").get<std::vector<int>>();require(prefix.size()==514,"Expected student 514-position prefix");
        auto config=readJson(root+"/student/config.json");config["base_dir"]=root+"/student/";config["thread_num"]=4;config["precision"]="low";
        config["memory"]="low";config["power"]="high";config["reuse_kv"]=false;config["attention_mode"]=8;config["cosyvoice_kleidiai"]=false;
        writePublicJson(root+"/student/comparison-config.json",config);auto tick=Clock::now();
        trace.mark("student-lm-loading");PublicLlm model(Llm::createLLM(root+"/student/comparison-config.json"),&Llm::destroy);require(model&&model->load(),"Student load failed");
        const double load=seconds(tick);model->reset();model->generate_init();trace.mark("student-lm-loaded");std::vector<int> codes;double prefill=0,decode=0;
        const int offset=metadata["music_offset"].get<int>();std::mt19937 random(2307);
        for(int i=0;i<frames;++i){require(seconds(trace.start)<660,"Student LM deadline");trace.gate(1200);tick=Clock::now();auto logits=publicLogits(*model,i?std::vector<int>{offset+codes.back()}:prefix,i==0);
            bool recent[16384]={false};for(size_t k=codes.size()>50?codes.size()-50:0;k<codes.size();++k)recent[codes[k]]=true;
            for(int k=0;k<16384;++k)if(recent[k])logits[k]/=1.1f;
            std::vector<int> indices(16384);std::iota(indices.begin(),indices.end(),0);
            std::partial_sort(indices.begin(),indices.begin()+50,indices.end(),[&](int a,int b){return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);});
            std::vector<double> weights(50);for(int k=0;k<50;++k)weights[k]=std::exp((double(logits[indices[k]])-logits[indices[0]])/.9);
            std::discrete_distribution<int> distribution(weights.begin(),weights.end());codes.push_back(indices[distribution(random)]);
            double duration=seconds(tick);if(i==0)prefill=duration;else decode+=duration;
            if(!i||codes.size()%25==0)trace.mark("real-phone-music-codes-"+std::to_string(codes.size()));
        }
        report["lm"]={{"loadSeconds",load},{"prefillSeconds",prefill},{"decodeSeconds",decode},{"decodeTokensPerSecond",(frames-1)/decode},{"prefixPositions",514},{"frames",codes.size()},
            {"finalKvPositions",model->getCurrentHistory()},{"eosSuppressedForFixedLengthDiagnostic",true},{"precision","low"},{"threads",4}};
        model.reset();report["lmStageSeconds"]=seconds(trace.start);trace.mark("student-lm-destroyed-before-codec");
        Json source={{"source","untrained 0.6B diagnostic student"},{"diagnosticOnly",true},{"lmDevice","Harmony phone CPU"},{"codes",codes},{"lmGuidance",1},
            {"progressPath",trace.sink}};writePublicJson(root+"/comparison-codes.json",source);
        std::string codePath=root+"/comparison-codes.json",steps="2",output="/dev/null",name="small";char* args[]={name.data(),const_cast<char*>(root.c_str()),codePath.data(),steps.data(),output.data()};Json codec;
        const int code=LevoPublicCodecMain(5,args,&codec);report["codec"]=codec;require(code==0,codec.value("error",std::string("Small codec failed")));
        report["validPcmSeconds"]=frames/25.;report["durationSeconds"]=frames/25.;report["pipelineSecondsThroughPcm"]=seconds(trace.start);report["pipelineRtfThroughPcm"]=seconds(trace.start)/(frames/25.);
        report["success"]=true;report["audioFileGenerated"]=false;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();return report;
}

std::string LevoRunComparableChain(const std::string& root,const std::string& variant){
    ComparisonMemory memory;Json result;const bool lowPublic=variant=="large-2-120-low";
    const bool ninety=variant=="large-2-90"||variant=="small-2-90";
    const bool longer=variant=="large-2-120"||variant=="small-2-120"||lowPublic||ninety;
    const bool small=variant=="small-2"||variant=="small-2-120"||variant=="small-2-90";const int frames=ninety?2250:(longer?3000:750);
    try{
        require(variant=="large-2"||variant=="small-2"||longer,"Invalid comparison variant");
        if(small)result=comparisonStudent(root+"/small",frames);
        else{auto input=readJson(root+"/input.json");result=musicGenerate(root+"/public",input["lyrics"],input["style"],frames,2,longer,lowPublic);
            result["variant"]="large";result["lmStageSeconds"]=result.contains("lm")?result["lm"]["totalLmSeconds"]:Json(nullptr);
            result["validPcmSeconds"]=result.value("durationSeconds",0.0);result["pipelineRtfThroughPcm"]=result.value("wholePhoneRtf",0.0);
        }
        result["memory"]=memory.finish();
        if(result.contains("sampledPeakRssKiB"))result["memory"]["peakRssKiB"]=std::max(result["memory"]["peakRssKiB"].get<uint64_t>(),result["sampledPeakRssKiB"].get<uint64_t>());
        if(result.value("success",false)){
            const double duration=result["durationSeconds"];double dit=0,vae=0;for(const auto& chunk:result["codec"]["chunks"]){dit+=chunk["ditSeconds"].get<double>();vae+=chunk["vaeSeconds"].get<double>();}
            const double llm=result["lmStageSeconds"],total=small?result["pipelineSecondsThroughPcm"].get<double>():result["totalSeconds"].get<double>();
            result["stageRtf"]={{"llm",llm/duration},{"dit",dit/duration},{"vae",vae/duration},{"other",(total-llm-dit-vae)/duration},{"total",total/duration}};
            result["stageSeconds"]={{"llm",llm},{"dit",dit},{"vae",vae},{"other",total-llm-dit-vae},{"total",total}};
            result["stageSeconds"]["prefill"]=result["lm"]["prefillSeconds"];result["stageSeconds"]["decode"]=result["lm"]["decodeSeconds"];
            result["stageRtf"]["prefill"]=result["lm"]["prefillSeconds"].get<double>()/duration;
            result["stageRtf"]["decode"]=result["lm"]["decodeSeconds"].get<double>()/duration;
        }
    }catch(const std::exception& e){result={{"success",false},{"error",e.what()},{"memory",memory.finish()}};}
    result["lifetimePolicy"]="Serial LLM complete and destroy, then one DiT and one VAE session reused for all chunks";
    result["quantization"]={{"lmBodyWeightBits",4},{"lmHeadWeightBits",8},{"inputEmbeddingBits",16},{"ditWeightBits",8},{"vaeWeightBits",8},{"lmGuidance",1},{"ditGuidance",1.5},
        {"lmPrecision",small||lowPublic?"Low":"High"},{"ditPrecision","High"},{"vaePrecision","Low"},{"lmThreads",4},{"codecThreads",6},
        {"activationQuantization","No requested static A4"},{"kvQuantization","Default runtime cache, no requested KV4"},{"samplingSeed",2307},{"temperature",.9},{"topK",50},{"repetitionPenalty",1.1}};
    result["requestedDurationSeconds"]=frames/25.;result["requestedMusicCodes"]=frames;result["fixedLengthBenchmark"]=longer;
    result["comparisonScope"]="Same App, requested"+std::to_string(frames/25)+"seconds, 2-step CFG DiT and same pretrained VAE; model sizes, condition contracts and training status differ";
    return result.dump();
}
