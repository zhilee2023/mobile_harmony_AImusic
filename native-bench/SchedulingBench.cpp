// Standalone experiments reuse the production numerical chain without changing
// the installed signed App. File transfer / host parsing excluded from RTF.
#define LEVO_CHAIN_LIBRARY
#include "ChainBench.cpp"

static std::string kvProbe(const std::string& dir,const std::string& requestedMode,int threads){
  const bool highPower=requestedMode=="persistent-high";
  const int attentionMode=requestedMode=="persistent-kv4"?14:requestedMode=="persistent-kv8"?10:8;
  const std::string mode=requestedMode.rfind("persistent",0)==0?"persistent":requestedMode;
  Trace trace;trace.sink=dir+"/last-kv-progress.jsonl";{std::ofstream out(trace.sink);}
  Json result={{"success",false},{"scope","Ten-window LLM-only teacher-forced numerical benchmark; identical fixed codec history, untrained music head; not end-to-end RTF or music quality"},
    {"mode",requestedMode},{"powerHint",highPower?"High":"Normal"},{"threads",threads},{"chunks",10},{"newCodes",1984},{"prefixTokens",514},
    {"usesKvWithinChunk",true},{"sameFullConditioning",mode!="window64"},{"sampleIntervalMs",20}};
  result["attentionMode"]=attentionMode;result["kvBits"]=attentionMode==14?4:attentionMode==10?8:16;result["activationBitsOverrideApplied"]=false;
  try{
    require(mode=="persistent"||mode=="full-reprefill"||mode=="window64","Invalid KV mode");
    require(threads==4||threads==6,"Invalid thread count");trace.gate(2048);
    auto metadata=readJson(dir+"/student/diagnostic-student.json");int offset=metadata["music_offset"].get<int>();
    auto prefix=readJson(dir+"/student/prefix.json").get<std::vector<int>>();require(prefix.size()==514,"Wrong prefix");
    auto config=readJson(dir+"/student/config.json");config["base_dir"]=dir+"/student/";config["llm_config"]="llm_config.json";config["embedding_file"]="embeddings_bf16.bin";config["reuse_kv"]=false;config["thread_num"]=threads;
    config["power"]=highPower?"high":"normal";
    config["attention_mode"]=attentionMode;
    std::string path=dir+"/student/config-kv-probe.json";{std::ofstream out(path);out<<config.dump(2);}
    auto tick=Clock::now();std::unique_ptr<Llm,decltype(&Llm::destroy)> llm(Llm::createLLM(path),&Llm::destroy);require(llm&&llm->load(),"LLM load failed");result["loadSeconds"]=seconds(tick);trace.mark("kv-llm-loaded");
    llm->reset();llm->generate_init();int generated=0;double allPrefill=0,allDecode=0,error=0,referenceEnergy=0;
    Json chunks=Json::array();int compared=0;size_t maximumKv=0;
    auto code=[&](int i){return offset+(i*37+2307)%16384;};
    for(int chunk=0;chunk<10;++chunk){
      const int count=chunk?192:256;double prefillTime=0,decodeTime=0;size_t prefillCount=0;
      for(int i=0;i<count;++i){
        trace.gate();require(seconds(trace.start)<360,"Bounded KV experiment timed out");
        std::vector<int> input;bool prefill=false;
        if(generated==0||(chunk&&i==0&&mode!="persistent")){
          if(generated){llm->reset();llm->generate_init();}
          input=prefix;int start=mode=="window64"?std::max(0,generated-64):0;
          for(int j=start;j<generated;++j)input.push_back(code(j));prefill=true;prefillCount=input.size();
        }else input={code(generated-1)};
        tick=Clock::now();auto logits=llm->forward(input,prefill);require(logits.get()&&logits->getInfo(),"LLM forward failed");
        const int size=logits->getInfo()->size;require(size>=16385,"Wrong logit size");const float* data=logits->readMap<float>();require(data,"LLM sync failed");data+=size-16385;
        for(int j=0;j<16385;++j)require(std::isfinite(data[j]),"Non-finite logits");
        double duration=seconds(tick);if(prefill)prefillTime+=duration;else decodeTime+=duration;
        maximumKv=std::max(maximumKv,llm->getCurrentHistory());
        if(i==0||i==count-1){
          auto refPath=dir+"/kv-logits-"+std::to_string(chunk)+"-"+(i==0?"first":"last")+".f32";
          if(requestedMode=="persistent"&&threads==4){std::ofstream out(refPath,std::ios::binary);out.write(reinterpret_cast<const char*>(data),16385*4);require(out.good(),"Cannot save cached reference");}
          else {auto ref=read(refPath,16385);for(int j=0;j<16385;++j){double d=data[j]-ref[j];error+=d*d;referenceEnergy+=double(ref[j])*ref[j];}++compared;}
        }
        ++generated;
      }
      allPrefill+=prefillTime;allDecode+=decodeTime;trace.mark("kv-chunk-"+std::to_string(chunk+1));
      chunks.push_back({{"chunk",chunk+1},{"prefillTokens",prefillCount},{"prefillSeconds",prefillTime},{"decodeSeconds",decodeTime},{"kvPositions",llm->getCurrentHistory()},{"rssKiB",resident()}});
    }
    result["forwardSeconds"]=allPrefill+allDecode;result["prefillSeconds"]=allPrefill;result["decodeSeconds"]=allDecode;
    result["decodeTokensPerSecond"]=(1984-(mode=="persistent"?1:10))/allDecode;result["maximumKvPositions"]=maximumKv;
    result["sampledLogitRelativeL2VsPersistent4"]=compared?Json(std::sqrt(error/std::max(referenceEnergy,1e-30))):Json(nullptr);
    result["logitComparisons"]=compared;result["perChunk"]=chunks;result["success"]=true;
  }catch(const std::exception& e){result["error"]=e.what();trace.mark("kv-stopped");}
  trace.sample();result["sampledPeakRssKiB"]=trace.maxRss.load();result["minimumSystemAvailableKiB"]=trace.minAvailable.load();result["totalSeconds"]=seconds(trace.start);result["stages"]=trace.stages;return result.dump();
}

int main(int argc,char** argv){
  if(argc!=5){std::cerr<<"Usage: root kv:persistent|kv:persistent-high|kv:full-reprefill|kv:window64|chain:eager|chain:overlap|chain:compute dit|small-dit threads4|6\n";return 2;}
  const std::string mode=argv[2];std::string report;
  try{const int threads=std::stoi(argv[4]);if(mode.rfind("kv:",0)==0)report=kvProbe(argv[1],mode.substr(3),threads);
    else{require(mode=="chain:eager"||mode=="chain:overlap"||mode=="chain:compute","Invalid mode");report=LevoRunChunks(argv[1],10,argv[3],mode=="chain:overlap",threads,mode=="chain:compute");}
    std::cout<<"LEVO_SCHEDULING_REPORT "<<report<<std::endl;return Json::parse(report)["success"].get<bool>()?0:1;
  }catch(const std::exception& e){std::cout<<"LEVO_SCHEDULING_REPORT "<<Json({{"success",false},{"error",e.what()}}).dump()<<std::endl;return 1;}
}
