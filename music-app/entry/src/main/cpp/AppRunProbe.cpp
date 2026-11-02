#include "MusicRuntime.hpp"
int main(int argc,char** argv){
    if(argc!=3)return 2;auto input=readJson(argv[2]);auto report=musicGenerate(argv[1],input["lyrics"],input["style"],375);
    std::cout<<"LEVO_MUSIC_APP_CORE_REPORT "<<report.dump()<<std::endl;return report["success"].get<bool>()?0:1;
}
