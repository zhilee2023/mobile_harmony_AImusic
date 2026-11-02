#pragma once
#include "RuntimeSupport.hpp"
#include <cctype>

namespace LevoBundle {
inline bool exists(const std::string& root){return std::filesystem::is_regular_file(root+"/model-bundle.json");}
inline bool validId(const std::string& id){
    if(id.empty()||id.size()>80||id.find("..")!=std::string::npos)return false;
    for(unsigned char c:id)if(!std::isalnum(c)&&c!='-'&&c!='_'&&c!='.')return false;
    return std::isalnum(static_cast<unsigned char>(id.front()));
}
inline std::string path(const std::string& root,const std::string& relative){
    std::filesystem::path value(relative);
    require(!relative.empty()&&!value.is_absolute()&&relative.find('\\')==std::string::npos&&relative.find(':')==std::string::npos,"Invalid model-bundle path");
    for(const auto& part:value)require(part!=".."&&part!=".","Model path escapes bundle");
    auto base=std::filesystem::weakly_canonical(root),target=std::filesystem::weakly_canonical(base/value);
    auto inside=target.lexically_relative(base);require(!inside.empty()&&*inside.begin()!="..","Model symlink escapes bundle");
    return target.string();
}
inline Json load(const std::string& root){
    auto manifest=readJson(root+"/model-bundle.json");
    require(manifest.value("schemaVersion",0)==1&&validId(manifest.value("modelId",std::string())),"Invalid model bundle identity/schema");
    const auto profile=manifest.value("profile",std::string());const auto family=manifest.value("family",std::string());
    require((profile=="public-mixed-v2"&&family=="public")||(profile=="student-mixed-v1"&&family=="small"),"Unsupported model bundle architecture");
    require(manifest.at("modules").at("llm").at("path")==std::string(family=="small"?"student/llm.mnn":"lm/llm.mnn"),"LLM graph and companion directory must match the family adapter");
    const auto& c=manifest.at("contract");
    require(c.at("id")=="levo-mixed-1rvq-25hz-64ch-48k-v1"&&c.at("vocabulary")==16384&&c.at("frameRate")==25&&c.at("windowFrames")==256&&
        c.at("contextFrames")==64&&c.at("hopFrames")==192&&c.at("conditionWidth")==1024&&c.at("ditInputWidth")==1200&&c.at("latentChannels")==64&&
        c.at("sampleRate")==48000&&c.at("audioChannels")==2&&c.at("samplesPerLatent")==1920,"Incompatible music codec contract");
    for(const char* role:{"llm","dit","vae"}){
        const auto& module=manifest.at("modules").at(role);path(root,module.at("path"));
        const auto& policy=module.at("runtime");int threads=policy.at("threads");std::string precision=policy.at("precision");
        require(threads>=1&&threads<=16&&(precision=="high"||precision=="low"),"Invalid module runtime policy");
    }
    require(manifest.at("runtime").value("backend",std::string("cpu"))=="cpu","This bundle runtime supports CPU; NPU/GPU remain experiments");
    return manifest;
}
inline std::string modulePath(const std::string& root,const char* role,const std::string& fallback){
    return exists(root)?path(root,load(root).at("modules").at(role).at("path")):root+"/"+fallback;
}
inline Json policy(const std::string& root,const char* role,const char* precision,int threads){
    return exists(root)?load(root).at("modules").at(role).at("runtime"):Json{{"precision",precision},{"threads",threads}};
}
inline Json weightBits(const std::string& root,const char* role,int fallback){
    return exists(root)?load(root).at("modules").at(role).at("recipe").value("weightBits",Json(nullptr)):Json(fallback);
}
inline std::string comparisonRoot(const std::string& root,const char* family){
    if(!exists(root))return root+"/"+family;
    require(load(root).at("family")==family,"Selected bundle belongs to the other comparison family");return root;
}
inline Json assets(const std::string& root,const Json& legacy){return exists(root)?load(root).at("assets"):legacy;}
inline Json inspect(const std::string& root,const Json& legacy){
    auto files=assets(root,legacy);Json missing=Json::array();uint64_t total=0;int ready=0;
    for(const auto& item:files){const std::string relative=item.at("relative");auto file=path(root,relative);uint64_t expected=item.at("bytes");total+=expected;
        if(std::filesystem::is_regular_file(file)&&std::filesystem::file_size(file)==expected)++ready;else missing.push_back(relative);}
    Json result={{"ready",missing.empty()},{"count",ready},{"total",files.size()},{"bytes",total},{"missing",missing},{"integrityMode","size-at-start; SHA256 verified by deployment tool"}};
    if(exists(root)){auto m=load(root);result["modelId"]=m["modelId"];result["profile"]=m["profile"];result["musicTrainedDeclared"]=m["musicTrained"];}
    return result;
}
}
