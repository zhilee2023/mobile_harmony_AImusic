#include "../runtime/include/levo/experimental/StreamLab.hpp"
int main(int argc,char** argv){
    if(argc<5){std::cerr<<"appRoot choice frames transport|mnn [stop-after-first]\n";return 2;}
    std::atomic<bool> done{false};std::thread stopper;
    const bool stop=argc==6&&std::string(argv[5])=="stop-after-first";
    if(stop)stopper=std::thread([&]{while(!done){if(LevoStreamLab::snapshot().value("generatedSamples",0)>0){LevoStreamLab::cancelled=true;break;}std::this_thread::sleep_for(std::chrono::milliseconds(10));}});
    Json request=argc==6&&!stop?readJson(argv[5]):Json::object();
    auto result=LevoStreamLab::run(argv[1],std::stoi(argv[2]),std::stoi(argv[3]),std::string(argv[4])=="transport",request.value("lyrics",std::string()),request.value("npy",std::string()),request);
    done=true;if(stopper.joinable())stopper.join();std::cout<<"STREAM_LAB_REPORT "<<result.dump()<<std::endl;
    return result["status"]=="completed"||result["status"]=="stopped"?0:1;
}
