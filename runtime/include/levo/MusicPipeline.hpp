#pragma once
#include "MixedLm.hpp"
#include "Conditioner.hpp"
#include <filesystem>

namespace LevoMusic {
inline Json checkAssets(const std::string& root,const Json& legacyAssets){
    return LevoBundle::inspect(root,legacyAssets);
}
inline Json generate(const std::string& root,const std::string& lyrics,const std::string& style,int frames,const Json& legacyAssets,int ditSteps=2,bool fixedLengthDiagnostic=false,bool lowLmPrecision=false){
    PublicPhoneTrace trace;trace.sink=root+"/progress.jsonl";Json report={{"success",false},{"wholePhoneChain",true},{"ditSteps",ditSteps},{"lmGuidance",1},{"ditGuidance",1.5}};
    std::ofstream(trace.sink).close();levoPublicCancelled=false;
    try{
        require(checkAssets(root,legacyAssets)["ready"],"请先导入完整的模型文件");require(frames==375||frames==750||frames==1500||frames==2250||frames==3000,"Unsupported audio duration");require(ditSteps>=2&&ditSteps<=20,"Unsupported DiT steps");
        if(LevoBundle::exists(root)){auto bundle=LevoBundle::load(root);require(bundle["family"]=="public"&&bundle.value("musicTrained",false),"歌曲生成需要已训练的公开Mixed模型包；原型请用研究入口");report["modelBundle"]=bundle["modelId"];}
        auto lmPolicy=LevoBundle::policy(root,"llm",lowLmPrecision?"low":"high",4);
        trace.gate(2600);trace.mark("preparing-current-lyrics-style");
        report["condition"]=prepareMusicPrefix(root,lyrics,style);
        Json config={{"base_dir",root+"/lm/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},{"llm_config","llm_config.json"},
            {"tokenizer_file","tokenizer.mtok"},{"embedding_file","generation-embeddings-bf16.bin"},{"backend_type","cpu"},
            {"thread_num",lmPolicy["threads"]},{"precision",lmPolicy["precision"]},{"memory","low"},{"power","high"},{"reuse_kv",false},{"attention_mode",8},
            {"levo_lm_guidance",1},{"levo_force_music_frames",fixedLengthDiagnostic},{"levo_run_id",std::to_string(std::chrono::system_clock::now().time_since_epoch().count())},{"levo_progress_path",trace.sink}};
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

} // namespace LevoMusic
