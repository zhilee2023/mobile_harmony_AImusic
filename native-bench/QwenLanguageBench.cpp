// Standalone original language model benchmark; no App or music-model changes.
#define LEVO_CHAIN_LIBRARY
#include "ChainBench.cpp"
#include "core/TensorUtils.hpp"

int main(int argc, char** argv) {
  if(argc!=6&&argc!=7){std::cerr<<"root variant threads attentionMode dynamicOption [low|high]\n";return 2;}
  Trace trace;
  Json report={{"success",false},{"execution","CLI, original Qwen language head; not music RTF"},
               {"packedActivation4KernelClaimed",false},{"sampleIntervalMs",20}};
  try {
    const std::string root=argv[1],variant=argv[2];
    const int threads=std::stoi(argv[3]),attention=std::stoi(argv[4]),dynamic=std::stoi(argv[5]);
    const std::string precision=argc==7?argv[6]:"low";
    require(precision=="low"||precision=="high","Unknown precision");report["precision"]=precision;
    require(variant=="w4"||variant=="smooth-a4"||variant=="smooth-a8"||variant=="smooth-noact","Unknown model variant");
    require((threads==4||threads==6)&&(attention==8||attention==14)&&(dynamic==0||dynamic==8),"Invalid benchmark settings");
    auto fixture=readJson(root+"/fixture/fixture.json");
    const auto prefix=fixture["prefix"].get<std::vector<int>>(),teacher=fixture["teacherTokens"].get<std::vector<int>>();
    const auto snapshots=fixture["snapshotSteps"].get<std::vector<int>>();
    require(prefix.size()==512&&teacher.size()==128,"Unexpected language fixture");
    report["variant"]=variant;report["threads"]=threads;report["attentionMode"]=attention;
    report["dynamicOption"]=dynamic;report["kvBits"]=attention==14?4:16;
    report["modelManifest"]=readJson(root+"/"+variant+"/language-probe.json");
    trace.gate(2048);
    auto config=readJson(root+"/"+variant+"/config.json");
    config["base_dir"]=root+"/"+variant+"/";config["llm_config"]="llm_config.json";
    config["embedding_file"]="embeddings_bf16.bin";config["reuse_kv"]=false;
    config["thread_num"]=threads;config["attention_mode"]=attention;config["dynamic_option"]=dynamic;
    config["power"]="normal";config["precision"]=precision;config["memory"]="low";
    const auto configPath=root+"/"+variant+"/run-config.json";
    {std::ofstream file(configPath);file<<config.dump(2);}
    auto tick=Clock::now();
    std::unique_ptr<Llm,decltype(&Llm::destroy)> llm(Llm::createLLM(configPath),&Llm::destroy);
    require(llm&&llm->load(),"Original language model load failed");report["loadSeconds"]=seconds(tick);
    trace.mark("language-loaded");
    Json rounds=Json::array();
    for(int round=0;round<2;++round){
      llm->reset();llm->generate_init();double prefill=0,decode=0,error=0,energy=0,nll=0;int agree=0;
      Json greedy=Json::array();
      for(int i=0;i<128;++i){
        trace.gate();require(seconds(trace.start)<240,"Language probe deadline");
        tick=Clock::now();auto output=llm->forward(i==0?prefix:std::vector<int>{teacher[i-1]},i==0);
        require(output.get()&&output->getInfo(),"Language model forward failed");
        const int size=output->getInfo()->size;require(size>=151936,"Unexpected full language head shape");
        const float* logits=output->readMap<float>();require(logits,"Language logit synchronization failed");logits+=size-151936;
        for(int j=0;j<151936;++j)require(std::isfinite(logits[j]),"Non-finite language logit");
        const double elapsed=seconds(tick);if(i==0)prefill=elapsed;else decode+=elapsed;
        int best=std::max_element(logits,logits+151936)-logits;greedy.push_back(best);agree+=(best==teacher[i]);
        const double maximum=logits[best];double normalizer=0;
        for(int j=0;j<151936;++j)normalizer+=std::exp(double(logits[j])-maximum);
        nll+=std::log(normalizer)+maximum-logits[teacher[i]];
        if(std::find(snapshots.begin(),snapshots.end(),i)!=snapshots.end()){
          auto ref=read(root+"/fixture/logits-"+std::to_string(i)+".f32",151936);
          for(int j=0;j<151936;++j){const double d=logits[j]-ref[j];error+=d*d;energy+=double(ref[j])*ref[j];}
        }
      }
      rounds.push_back({{"round",round+1},{"prefillTokens",512},{"decodeTokens",127},
        {"prefillSeconds",prefill},{"prefillTokensPerSecond",512/prefill},{"decodeSeconds",decode},
        {"decodeTokensPerSecond",127/decode},{"sampledLogitRelativeL2VsFp32",std::sqrt(error/std::max(energy,1e-30))},
        {"teacherGreedyAgreement",double(agree)/128},{"referenceContinuationMeanNll",nll/128},
        {"teacherForcedArgmaxIds",greedy},{"kvPositions",llm->getCurrentHistory()}});
      trace.mark("language-round-"+std::to_string(round+1));
    }
    report["rounds"]=rounds;
    // Separate greedy generation for inspecting actual text; excluded above timings.
    llm->reset();llm->generate_init();Json generated=Json::array();std::vector<int> input=prefix;
    for(int i=0;i<48;++i){
      trace.gate();require(seconds(trace.start)<240,"Language generation deadline");
      auto output=llm->forward(input,i==0);require(output.get()&&output->getInfo(),"Greedy forward failed");
      const int size=output->getInfo()->size;require(size>=151936,"Wrong greedy vocabulary");
      const float* p=output->readMap<float>();require(p,"Greedy read failed");p+=size-151936;
      for(int j=0;j<151936;++j)require(std::isfinite(p[j]),"Non-finite greedy logit");
      int id=std::max_element(p,p+151936)-p;generated.push_back(id);if(llm->is_stop(id))break;input={id};
    }
    report["greedyOutputIds"]=generated;report["success"]=true;
  }catch(const std::exception& e){report["error"]=e.what();}
  trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();
  report["totalSeconds"]=seconds(trace.start);report["stages"]=trace.stages;
  std::cout<<"LEVO_LANGUAGE_REPORT "<<report.dump()<<std::endl;return report["success"].get<bool>()?0:1;
}
