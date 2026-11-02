#pragma once
#define LEVO_PUBLIC_PHONE_LIBRARY
#include "PublicMixedPhone.cpp"
#include "MusicConditioner.hpp"
#include "MusicAssets.hpp"
#include <filesystem>

static Json musicCheckAssets(const std::string& root){
    Json missing=Json::array();uint64_t total=0;int ready=0;
    for(const auto& item:musicAssetInventory()){
        struct stat s{};uint64_t expected=item["bytes"].get<uint64_t>();total+=expected;
        if(stat((root+"/"+item["relative"].get<std::string>()).c_str(),&s)==0&&uint64_t(s.st_size)==expected)++ready;
        else missing.push_back(item["name"]);
    }
    return {{"ready",missing.empty()},{"count",ready},{"total",musicAssetInventory().size()},{"bytes",total},{"missing",missing}};
}
static Json musicGenerate(const std::string& root,const std::string& lyrics,const std::string& style,int frames,int ditSteps=4){
    PublicPhoneTrace trace;trace.sink=root+"/progress.jsonl";Json report={{"success",false},{"wholePhoneChain",true},{"ditSteps",ditSteps},{"lmGuidance",1},{"ditGuidance",1.5}};
    std::ofstream(trace.sink).close();levoPublicCancelled=false;
    try{
        require(musicCheckAssets(root)["ready"],"请先导入完整的模型文件");require(frames==375||frames==750||frames==1500||frames==2250||frames==3000,"Unsupported audio duration");require(ditSteps==2||ditSteps==4,"Unsupported DiT steps");
        trace.gate(2600);trace.mark("preparing-current-lyrics-style");
        report["condition"]=prepareMusicPrefix(root,lyrics,style);
        Json config={{"base_dir",root+"/lm/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},{"llm_config","llm_config.json"},
            {"tokenizer_file","tokenizer.mtok"},{"embedding_file","generation-embeddings-bf16.bin"},{"backend_type","cpu"},
            {"thread_num",4},{"precision","high"},{"memory","low"},{"power","high"},{"reuse_kv",false},{"attention_mode",8},
            {"levo_lm_guidance",1},{"levo_run_id",std::to_string(std::chrono::system_clock::now().time_since_epoch().count())},{"levo_progress_path",trace.sink}};
        writePublicJson(root+"/lm/config-phone.json",config);
        report["lm"]=generatePublicTokens(root,frames,trace,false);trace.gate(1500);
        std::filesystem::create_directories(root+"/songs");std::string wave=root+"/songs/"+config["levo_run_id"].get<std::string>()+".wav";
        std::string codes=root+"/phone-generated-codes.json",steps=std::to_string(ditSteps),label="LevoMusic";
        char* args[]={label.data(),const_cast<char*>(root.c_str()),codes.data(),steps.data(),wave.data()};Json codec;
        const int error=LevoPublicCodecMain(5,args,&codec);report["codec"]=codec;require(error==0,codec.value("error",std::string("Audio decode failed")));
        report["wave"]=wave;report["durationSeconds"]=codec["durationSeconds"];report["totalSeconds"]=seconds(trace.start);
        report["wholePhoneRtf"]=seconds(trace.start)/codec["durationSeconds"].get<double>();report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();report["cancelled"]=levoPublicCancelled.load();}
    trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();
    report["totalSeconds"]=seconds(trace.start);writePublicJson(root+"/last-music-result.json",report);return report;
}
