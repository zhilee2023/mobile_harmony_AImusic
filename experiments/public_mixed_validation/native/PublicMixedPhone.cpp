// Real pretrained Mixed LM sampling and four-step Codec on one Harmony phone.
#define LEVO_PUBLIC_CODEC_LIBRARY
#include "PublicMixedCodec.cpp"
#include <numeric>

using PublicLlm=std::unique_ptr<Llm,decltype(&Llm::destroy)>;

// The public ~3B Mixed LM uses two CFG contexts. Keep the older student
// diagnostic's 5 GiB limit unchanged; this test has a separate bounded budget.
struct PublicPhoneTrace:Trace{
    void gate(long mib=1024){
        require(!levoPublicCancelled.load(),"Generation cancelled");
        sample();require(available()>=std::max(mib,1200L)*1024,"Public LM system-available memory gate");
        require(resident()<7L*1024*1024,"Public LM process RSS reached7GiB diagnostic stop");
    }
};

static void writePublicJson(const std::string& path,const Json& value){
    std::ofstream out(path);out<<value.dump(2);require(out.good(),"Cannot write report: "+path);
}

static std::vector<float> publicLogits(Llm& model,const std::vector<int>& ids,bool prefill){
    auto logits=model.forward(ids,prefill);
    require(logits.get()!=nullptr&&logits->getInfo()!=nullptr,"No Mixed logits");
    const auto count=logits->getInfo()->size;
    require(count>=16385,"Wrong Mixed logit shape");
    const float* data=logits->readMap<float>();require(data,"Mixed logits read failed");
    std::vector<float> values(data+count-16385,data+count);peak(values);return values;
}

static Json generatePublicTokens(const std::string& root,int frames,PublicPhoneTrace& trace,bool probeOnly){
    const auto start=Clock::now();Json config=readJson(root+"/lm/config-phone.json");
    require(config["thread_num"]==4&&(config["precision"]=="high"||config["precision"]=="low"),"Expected CPU High/Low / 4 threads");
    const double guidance=config.value("levo_lm_guidance",1.5);
    const bool fixedLength=config.value("levo_force_music_frames",false);
    require(guidance==1.0||guidance==1.5,"Controlled LM CFG choices are 1 / 1.5");
    auto load=[&](const std::string& name){
        trace.gate(1200);auto tick=Clock::now();
        trace.mark(name+"-loading");PublicLlm model(Llm::createLLM(root+"/lm/config-phone.json"),&Llm::destroy);
        require(model&&model->load(),"Mixed LM load failed: "+name);
        model->reset();model->generate_init();trace.mark(name+"-loaded");
        return std::make_pair(std::move(model),seconds(tick));
    };
    auto first=load("conditioned-lm");auto& conditional=first.first;
    Json checks=Json::array();int teacher=0;
    for(int i=0;i<2;++i){
        trace.gate(900);std::vector<int> ids;
        if(!i){ids.resize(32);std::iota(ids.begin(),ids.end(),16386+1906);}else ids={teacher};
        const auto values=publicLogits(*conditional,ids,i==0);
        auto reference=read(root+"/lm/mapping-"+std::to_string(i)+"-reference.f32",16385);
        const auto error=relative(values,reference);teacher=int(std::max_element(reference.begin(),reference.end())-reference.begin());
        const int top1=int(std::max_element(values.begin(),values.end())-values.begin());
        checks.push_back({{"phase",i?"decode":"prefill"},{"logitsL2VsOriginal",error},{"top1",top1},{"teacher",teacher}});
        require(error<.25,"Phone W4 LM probe exceeds bounded numerical tolerance");
        trace.mark(i?"lm-decode-probe-passed":"lm-prefill-probe-passed");
    }
    conditional->reset();conditional->generate_init();
    if(probeOnly){Json report={{"success",true},{"probeOnly",true},{"rows",checks},{"lmLoadSeconds",first.second}};conditional.reset();return report;}
    PublicLlm unconditional(nullptr,&Llm::destroy);double secondLoad=0;
    if(guidance>1){auto second=load("unconditional-lm");unconditional=std::move(second.first);secondLoad=second.second;}
    Json result={{"source","trained public LeVo v2-large primary Mixed LM"},{"lmDevice","Harmony phone CPU"},
        {"mixedOnly",true},{"randomMusicHead",false},{"bodyBits",4},{"headBits",8},{"lmGuidance",guidance},
        {"temperature",.9},{"topK",50},{"recentTokenWindow",50},{"seed",2307},{"prefixPositions",953},
        {"threads",4},{"precision",config["precision"]},{"maxFramesRequested",frames},{"probe",checks},
        {"lmLoadSeconds",first.second+secondLoad},{"samplingGenerator","C++ mt19937"},
        {"lmInstances",guidance>1?2:1},{"runId",config.value("levo_run_id",std::string("unspecified"))},
        {"progressPath",config.value("levo_progress_path",std::string())},{"eosSuppressedForFixedLengthDiagnostic",fixedLength}};
    std::vector<int> condPrefix(953),nullPrefix(953);std::iota(condPrefix.begin(),condPrefix.end(),16386);
    std::iota(nullPrefix.begin(),nullPrefix.end(),16386+953);
    std::mt19937 generator(2307);std::vector<int> codes;std::vector<double> timings;bool eos=false;double prefill=0;
    for(int i=0;i<frames;++i){
        require(seconds(start)<(fixedLength?1200:660),"LM deadline exceeded; partial tokens retained");trace.gate(800);
        auto tick=Clock::now();if(!i)trace.mark("conditioned-full-prefill-start");
        const auto left=publicLogits(*conditional,i?std::vector<int>{codes.back()}:condPrefix,i==0);
        std::vector<float> logits=left;
        if(unconditional){
            trace.gate(800);if(!i)trace.mark("unconditional-full-prefill-start");
            const auto right=publicLogits(*unconditional,i?std::vector<int>{codes.back()}:nullPrefix,i==0);
            for(size_t k=0;k<left.size();++k)logits[k]=right[k]+float(guidance)*(left[k]-right[k]);
        }
        bool recent[16385]={false};for(size_t k=codes.size()>50?codes.size()-50:0;k<codes.size();++k)recent[codes[k]]=true;
        for(size_t k=0;k<logits.size();++k)if(recent[k])logits[k]/=1.1f;
        if(fixedLength)logits[16384]=-1e30f;
        std::vector<int> indices(logits.size());std::iota(indices.begin(),indices.end(),0);
        std::partial_sort(indices.begin(),indices.begin()+50,indices.end(),[&](int a,int b){return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);});
        std::vector<double> weights(50);const double largest=logits[indices[0]];
        for(int k=0;k<50;++k)weights[k]=std::exp((double(logits[indices[k]])-largest)/.9);
        std::discrete_distribution<int> sample(weights.begin(),weights.end());int token=indices[sample(generator)];
        double elapsed=seconds(tick);timings.push_back(elapsed);if(!i)prefill=elapsed;
        if(token==16384){eos=true;break;}require(token>=0&&token<16384,"Invalid music code");codes.push_back(token);
        if(i==0||codes.size()%25==0||i+1==frames){
            result["codes"]=codes;result["frames"]=codes.size();result["audioSeconds"]=codes.size()/25.;
            result["prefillSeconds"]=prefill;result["totalLmSeconds"]=seconds(start);result["eos"]=eos;
            trace.sample();result["checkpointSampledPeakRssKiB"]=trace.maxRss.load();result["checkpointMinimumSystemAvailableKiB"]=trace.minAvailable.load();
            result["checkpointPid"]=getpid();
            writePublicJson(root+"/phone-generated-codes.partial.json",result);
            trace.mark("real-phone-music-codes-"+std::to_string(codes.size()));
        }
    }
    require(!codes.empty(),"EOS before first music token");
    result["codes"]=codes;result["frames"]=codes.size();result["audioSeconds"]=codes.size()/25.;result["eos"]=eos;
    result["prefillSeconds"]=prefill;result["tokenSeconds"]=timings;
    double decode=std::accumulate(timings.begin()+1,timings.end(),0.);
    result["decodeSeconds"]=decode;result["decodeTokensPerSecond"]=(timings.size()-1)/std::max(decode,1e-12);
    result["conditionedKvPositions"]=conditional->getCurrentHistory();result["unconditionalKvPositions"]=unconditional?unconditional->getCurrentHistory():0;
    result["lmSecondsThroughLastToken"]=seconds(start);result["lmCleanupCompleted"]=false;
    writePublicJson(root+"/phone-generated-codes.pre-destroy.json",result);trace.mark("public-lm-destroy-start");
    conditional.reset();unconditional.reset();result["lmCleanupCompleted"]=true;
    trace.mark("both-public-lms-destroyed-before-codec");result["totalLmSeconds"]=seconds(start);
    writePublicJson(root+"/phone-generated-codes.json",result);return result;
}

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
