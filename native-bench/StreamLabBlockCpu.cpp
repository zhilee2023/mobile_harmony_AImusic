// Same captured block/math as the portable OM; not full DiT timing.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
using namespace LevoStreamLab;
int main(int argc,char** argv){
    if(argc!=4)return 2;const std::string dir=argv[1],precision=argv[2],sink=argv[3];Json report={{"success",false},{"scope","One captured DiT block, expanded W8 source weights, CPU comparison to NPU FP16; not original full quantized DiT"},{"precision",precision}};Trace trace;
    try{require(precision=="high"||precision=="low","Unknown precision");require(!std::filesystem::exists(sink),"Preserve previous CPU result");
        auto input=read(dir+"/input.f32",2*1200*264),reference=read(dir+"/reference.f32",2*1200*256);auto tick=Clock::now();Graph model(dir+"/block.mnn",precision=="low",6);report["setupSeconds"]=seconds(tick);report["runs"]=Json::array();
        for(int i=0;i<4;++i){check();tick=Clock::now();auto result=model.run({{"inputs",input}},reference.size());auto elapsed=seconds(tick);report["runs"].push_back({{"mode",i?"warm":"first"},{"totalSeconds",elapsed},{"relativeL2",relative(result,reference)}});std::cout<<"BLOCK_CPU_RUN "<<elapsed<<std::endl;}
        report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;std::ofstream(sink)<<report.dump(2);std::cout<<"BLOCK_CPU_DONE "<<report.value("success",false)<<std::endl;return report["success"]==true?0:1;
}
