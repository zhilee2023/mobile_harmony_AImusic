#include "levo/operator_lab/Baseline.hpp"
int main(int argc,char** argv){
    if(argc!=4){std::cerr<<"APP_DATA_ROOT MODEL_0_OR_1 OPTIONS_JSON_FILE\n";return 2;}
    std::atomic<bool> done{false};std::thread watcher;
    try{
        LlmOperatorLab::busy=true;LlmOperatorLab::cancelled=false;
        watcher=std::thread([&]{std::string last;while(!done.load()){
            const auto p=LlmOperatorLab::poll();const auto text=Json({{"phase",p.value("phase",std::string("prepare"))},{"round",p.value("round",0)},{"decodeDone",p.value("decodeDone",0)}}).dump();
            if(text!=last){last=text;std::cout<<"OPERATOR_PROGRESS "<<text<<std::endl;}
            std::this_thread::sleep_for(std::chrono::milliseconds(500));}});
        const auto report=LlmOperatorLab::run(argv[1],std::stoi(argv[2]),readJson(argv[3]));
        done=true;watcher.join();
        LlmOperatorLab::busy=false;auto summary=report;summary.erase("operators");for(auto& row:summary["rounds"])row.erase("decodeTokenMs");
        std::cout<<"LLM_OPERATOR_DONE "<<summary.dump()<<std::endl;return report["status"]=="completed"?0:1;
    }catch(const std::exception& error){done=true;if(watcher.joinable())watcher.join();std::cerr<<error.what()<<std::endl;return 1;}
}
