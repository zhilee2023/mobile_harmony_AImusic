#define LEVO_PUBLIC_PHONE_LIBRARY
#include "PublicMixedPhone.cpp"
#include "MusicConditioner.hpp"
int main(int argc,char** argv){
    Json result={{"success",false}};
    try{
        require(argc==3,"root fixtures.json");const std::string root=argv[1];auto fixtures=readJson(argv[2]);Json rows=Json::array();
        for(const auto& test:fixtures){
            auto actual=prepareMusicPrefix(root,test["lyrics"],test["style"]);
            require(actual["lyricTokenIds"]==test["lyricIds"],"Phone lyric tokenizer differs from pinned HF tokenizer");
            require(actual["styleTokenIds"]==test["styleIds"],"Phone style tokenizer differs from pinned HF tokenizer");
            rows.push_back({{"lyricTokens",actual["lyricTokenIds"].size()},{"styleTokens",actual["styleTokenIds"].size()},{"seconds",actual["seconds"]}});
        }
        result["cases"]=rows;result["success"]=true;
    }catch(const std::exception& e){result["error"]=e.what();}
    std::cout<<"LEVO_CONDITION_REPORT "<<result.dump()<<std::endl;return result["success"].get<bool>()?0:1;
}
