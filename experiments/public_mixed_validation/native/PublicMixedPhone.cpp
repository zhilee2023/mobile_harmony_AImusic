#include "../../../runtime/include/levo/MixedLm.hpp"
#ifndef LEVO_PUBLIC_PHONE_LIBRARY
int main(int argc,char** argv){
    if(argc!=5){std::cerr<<"publicRoot codecRoot frames probe|generate\n";return 2;}
    PublicPhoneTrace trace;Json report={{"success",false},{"wholePhoneChain",true},{"randomMusicHead",false},
        {"lmDevice","Harmony phone CPU"},{"codecDevice","Harmony phone CPU"},{"ditSteps",4},
        {"lmProcessRssStopMiB",7168},{"lmMinimumAvailableMiB",1200}};
    const std::string root=argv[1];
    try{
        report["runId"]=readJson(root+"/lm/config-phone.json").value("levo_run_id",std::string("unspecified"));
        const int frames=std::stoi(argv[3]);require(frames>=25&&frames<=750,"Bounded real-music test");
        trace.gate(2600);trace.mark("whole-phone-real-mixed-start");
        const bool probe=std::string(argv[4])=="probe";auto tokens=generatePublicTokens(root,frames,trace,probe);report["lm"]=tokens;
        if(probe){report["probeOnly"]=true;report["success"]=true;}
        else{
            trace.gate(1500);std::string codes=root+"/phone-generated-codes.json",steps="4",wav=root+"/whole-phone-dit4.wav";
            char* codecArgs[]={argv[0],argv[2],codes.data(),steps.data(),wav.data()};Json codec;
            const int code=LevoPublicCodecMain(5,codecArgs,&codec);report["codec"]=codec;
            require(code==0,"Phone Codec failed; LM tokens retained");
            report["durationSeconds"]=codec["durationSeconds"];report["secondsThroughWav"]=seconds(trace.start);
            report["wholePhoneRtf"]=seconds(trace.start)/codec["durationSeconds"].get<double>();
            report["audioFileGenerated"]=true;report["wave"]=wav;report["success"]=true;
        }
    }catch(const std::exception& e){report["error"]=e.what();trace.mark("whole-phone-stopped");}
    trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();
    report["totalSeconds"]=seconds(trace.start);report["stages"]=trace.stages;
    writePublicJson(root+"/whole-phone-report.json",report);std::cout<<"LEVO_PUBLIC_PHONE_REPORT "<<report.dump()<<std::endl;
    return report["success"].get<bool>()?0:1;
}
#endif
