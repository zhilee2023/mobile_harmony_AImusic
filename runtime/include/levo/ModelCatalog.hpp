#pragma once
#include "ModelBundle.hpp"

namespace LevoBundle {
inline std::string activeMusicId(const std::string& files){
    if(!std::filesystem::is_regular_file(files+"/active-model.json"))return "builtin";
    auto id=readJson(files+"/active-model.json").value("id",std::string("builtin"));require(validId(id),"Invalid selected model ID");return id;
}
inline Json musicModels(const std::string& files,const Json& legacy){
    const auto active=activeMusicId(files);Json entries=Json::array();
    auto baseline=inspect(files+"/music-models",legacy);
    entries.push_back({{"id","builtin"},{"name","默认歌曲模型"},{"ready",baseline["ready"]},{"availableForMusic",true},{"legacy",true}});
    const std::string folder=files+"/model-bundles";
    if(std::filesystem::is_directory(folder))for(const auto& entry:std::filesystem::directory_iterator(folder)){
        const auto id=entry.path().filename().string();if(!entry.is_directory()||!validId(id)||id.find(".incoming-")!=std::string::npos)continue;
        try{auto manifest=load(entry.path().string());auto state=inspect(entry.path().string(),Json::array());
            require(manifest["modelId"]==id,"Bundle directory/identity mismatch");
            entries.push_back({{"id",id},{"name",manifest.value("displayName",id)},{"profile",manifest["profile"]},{"ready",state["ready"]},
                {"availableForMusic",manifest["family"]=="public"&&manifest.value("musicTrained",false)},{"qualityValidated",manifest.value("qualityValidated",false)}});
        }catch(const std::exception& error){entries.push_back({{"id",id},{"name",id},{"ready",false},{"availableForMusic",false},{"error",error.what()}});}
    }
    return {{"success",true},{"activeId",active},{"entries",entries}};
}
inline Json selectMusicModel(const std::string& files,const std::string& id,const Json& legacy){
    require(validId(id),"Invalid selected model ID");auto catalog=musicModels(files,legacy);bool found=false;
    for(const auto& row:catalog["entries"])if(row["id"]==id){require(row.value("availableForMusic",false),"此模型仍用于结构研究，尚不能用于歌曲生成");found=true;break;}
    require(found,"模型尚未导入，请先部署完整模型包");
    std::string temp=files+"/active-model.json.next";std::ofstream out(temp);out<<Json{{"id",id}}.dump();out.close();require(out.good(),"Cannot save selected model");
    std::filesystem::rename(temp,files+"/active-model.json");return {{"success",true},{"activeId",id}};
}
}
