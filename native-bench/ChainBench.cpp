#include "ChainBench.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <llm/llm.hpp>
#include <json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#ifdef LEVO_VAE_VECTOR_SIN
#include "VaeVectorSin.hpp"
#include "VaeSnakeRuntime.hpp"
#endif
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using MNN::Transformer::Llm;
static double seconds(Clock::time_point tick){return std::chrono::duration<double>(Clock::now()-tick).count();}
static long resident(){long a=0,b=0;std::ifstream("/proc/self/statm")>>a>>b;return b*sysconf(_SC_PAGESIZE)/1024;}
static long available(){std::ifstream in("/proc/meminfo");std::string line;while(std::getline(in,line))if(line.rfind("MemAvailable:",0)==0)return std::stol(line.substr(13));return -1;}
static void require(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);}
static Json readJson(const std::string& path){std::ifstream in(path);require(in.good(),"Missing JSON: "+path);Json value;in>>value;return value;}
static std::vector<float> read(const std::string& path,size_t count){std::ifstream in(path,std::ios::binary|std::ios::ate);require(in&&in.tellg()==std::streamoff(count*4),"Fixture size/read mismatch: "+path);std::vector<float> value(count);in.seekg(0);in.read(reinterpret_cast<char*>(value.data()),count*4);require(in.good(),"Short fixture read");return value;}
static double peak(const std::vector<float>& values){double result=0;for(float v:values){require(std::isfinite(v),"Non-finite tensor");result=std::max(result,std::abs(double(v)));}return result;}
static double relative(const std::vector<float>& a,const std::vector<float>& b){require(a.size()==b.size(),"Reference shape mismatch");double error=0,energy=0;for(size_t i=0;i<a.size();++i){double d=a[i]-b[i];error+=d*d;energy+=double(b[i])*b[i];}return std::sqrt(error/std::max(energy,1e-30));}
struct Trace{
  Clock::time_point start=Clock::now();std::atomic<bool> stop{false};std::atomic<long> maxRss{0},minAvailable{LONG_MAX};std::thread sampler;Json stages=Json::array();std::string sink;
  Trace(){sampler=std::thread([this]{while(!stop){sample();std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}
  ~Trace(){stop=true;sampler.join();}
  void sample(){long value=resident(),old=maxRss.load();while(value>old&&!maxRss.compare_exchange_weak(old,value)){}value=available();old=minAvailable.load();if(value>=0)while(value<old&&!minAvailable.compare_exchange_weak(old,value)){};}
  void mark(const std::string& name){sample();Json row={{"stage",name},{"seconds",seconds(start)},{"rssKiB",resident()},{"systemAvailableKiB",available()}};stages.push_back(row);if(!sink.empty()){std::ofstream out(sink,std::ios::app);out<<row.dump()<<'\n';}std::cout<<"LEVO_CHAIN_PROGRESS "<<row.dump()<<std::endl;}
  void gate(long mib=1024){sample();require(available()>=mib*1024,"System available below stage safety gate");require(resident()<5L*1024*1024,"Process RSS reached5GiB diagnostic stop");}
};
class Session{
  std::unique_ptr<MNN::Interpreter> net;MNN::Session* session=nullptr;
#ifdef LEVO_VAE_VECTOR_SIN
  std::unique_ptr<LevoVaeSnake::Runtime> snake;
  bool snakeValidated=false;double validationSeconds=0,validationL2=0;
#endif
public:
  Session(const std::string& model,bool low,int threads=6,bool snakeFusion=false){
    net.reset(MNN::Interpreter::createFromFile(model.c_str()));require(bool(net),"Cannot read MNN model");
    if(snakeFusion){require(low&&threads==6,"Snake runtime requires VAE CPU Low / 6 threads");net->setSessionMode(MNN::Interpreter::Session_Debug);}
    auto weight=model+".weight";net->setExternalFile(weight.c_str());net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
    MNN::BackendConfig backend;backend.precision=low?MNN::BackendConfig::Precision_Low:MNN::BackendConfig::Precision_High;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig cfg;cfg.type=MNN_FORWARD_CPU;cfg.numThread=threads;cfg.backendConfig=&backend;
    session=net->createSession(cfg);require(session,"Cannot create MNN session");net->resizeSession(session);net->releaseModel();
#ifdef LEVO_VAE_VECTOR_SIN
    if(snakeFusion)snake=std::make_unique<LevoVaeSnake::Runtime>(model,model.substr(0,model.find_last_of("/\\")+1)+"snake-coefficients.json");
#else
    require(!snakeFusion,"Snake runtime not compiled");
#endif
  }
  ~Session(){if(session)net->releaseSession(session);}
  std::vector<float> run(const std::map<std::string,std::vector<float>>& values,const std::string& output,size_t count){
    auto putInputs=[&](){for(const auto& pair:values){auto* input=net->getSessionInput(session,pair.first.c_str());require(input,"Missing model input: "+pair.first);MNN::Tensor host(input,MNN::Tensor::CAFFE,true);require(host.elementSize()==pair.second.size(),"Model input size mismatch");std::memcpy(host.host<float>(),pair.second.data(),pair.second.size()*4);require(input->copyFromHostTensor(&host),"Input copy failed");}};putInputs();
    std::vector<float> reference;
#ifdef LEVO_VAE_VECTOR_SIN
    // One real-input reference forward in this same VAE session validates App
    // integration. Its full cost is included in the first chunk and total RTF.
    if(snake&&!snakeValidated){auto tick=Clock::now();require(net->runSession(session)==MNN::NO_ERROR,"Snake reference forward failed");auto* tensor=net->getSessionOutput(session,output.c_str());require(tensor,"Missing reference PCM");MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(host.elementSize()==count&&tensor->copyToHostTensor(&host),"Snake reference copy failed");reference.assign(host.host<float>(),host.host<float>()+count);peak(reference);putInputs();validationSeconds=seconds(tick);}
    if(snake)snake->run(net.get(),session);else
#endif
    require(net->runSession(session)==MNN::NO_ERROR,"MNN forward failed");auto* tensor=net->getSessionOutput(session,output.c_str());require(tensor,"Missing model output");MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(host.elementSize()==count&&tensor->copyToHostTensor(&host),"Output shape/copy failed");std::vector<float> result(count);std::memcpy(result.data(),host.host<float>(),count*4);peak(result);
#ifdef LEVO_VAE_VECTOR_SIN
    if(snake&&!snakeValidated){validationL2=relative(result,reference);require(validationL2<.001,"Snake real-input PCM parity failed; L2="+std::to_string(validationL2));snakeValidated=true;}
#endif
    return result;
  }
  Json snakeStats()const{
#ifdef LEVO_VAE_VECTOR_SIN
    if(snake){auto stats=snake->stats();stats["firstRealInputParityPassed"]=snakeValidated;stats["firstRealInputPcmL2"]=validationL2;stats["includedReferenceValidationSeconds"]=validationSeconds;return stats;}
#endif
    return Json();
  }
};
std::string LevoRunChainBench(const std::string& dir,const std::string& requestedPolicy){
  bool w4=requestedPolicy.size()>3&&requestedPolicy.substr(requestedPolicy.size()-3)=="-w4";
  std::string policy=w4?requestedPolicy.substr(0,requestedPolicy.size()-3):requestedPolicy;
  const std::string ditName=w4?"full-w4.mnn":"full-w8.mnn",referenceName=w4?"w4-reference.f32":"w8-reference.f32";
  Trace trace;trace.sink=dir+"/last-chain-progress.jsonl";{std::ofstream clear(trace.sink);}
  Json result={{"success",false},{"flowAndMemoryPassed",false},{"policy",policy},{"ditWeightBits",w4?4:8},{"requestedPolicy",requestedPolicy},{"music_trained",false},{"real_song_generated",false},{"scope","Actual26-layer student with RANDOM music adapters -> complete39-layer quantized DiT TWO steps -> real CPU Low VAE; one diagnostic window, no numerical/perceptual parity claim"},{"sampleIntervalMs",20},{"ditSteps",2},{"frames",256},{"validPcmSeconds",10.24},{"llmThreads",4},{"codecThreads",6}};
  auto finish=[&](){trace.sample();result["stages"]=trace.stages;result["sampledPeakRssKiB"]=trace.maxRss.load();result["minimumSystemAvailableKiB"]=trace.minAvailable.load();result["totalSeconds"]=seconds(trace.start);return result.dump();};
  try{
    bool llmOnly=policy=="llm-short"||policy=="llm-long"||policy=="llm-kleidi";
    require(policy=="serial"||policy=="keep-llm"||policy=="all"||llmOnly,"Unknown lifetime policy");trace.gate(2048);trace.mark("start");
    auto metadata=readJson(dir+"/student/diagnostic-student.json");require(metadata["music_trained"]==false&&metadata["layers"]==26,"Expected diagnostic student");
    auto prefix=readJson(dir+"/student/prefix.json").get<std::vector<int>>();require(prefix.size()==514,"Expected512lyric+1style+BOS prefix");
    if(policy=="llm-short"){int bos=prefix.back();prefix.resize(137);prefix.push_back(bos);}
    auto config=readJson(dir+"/student/config.json");config["base_dir"]=dir+"/student/";config["llm_config"]="llm_config.json";config["embedding_file"]="embeddings_bf16.bin";config["reuse_kv"]=false;
    config["cosyvoice_kleidiai"]=(policy=="llm-kleidi");result["llmKleidiRequested"]=(policy=="llm-kleidi");
    std::string runtime=dir+"/student/config-runtime.json";{std::ofstream out(runtime);out<<config.dump(2);require(out.good(),"Cannot write local runtime config");}
    std::unique_ptr<Llm,decltype(&Llm::destroy)> llm(Llm::createLLM(runtime),&Llm::destroy);require(llm&&llm->load(),"LLM load failed");trace.mark("llm-loaded");trace.gate();
    std::unique_ptr<Session> dit,vae;
    if(policy=="all"){dit=std::make_unique<Session>(dir+"/dit/"+ditName,false);trace.mark("dit-loaded-alongside-llm");trace.gate();vae=std::make_unique<Session>(dir+"/vae/vae-256-fp32.mnn",true);trace.mark("all-three-sessions-loaded");trace.gate();}
    llm->reset();llm->generate_init();std::vector<int> codes;codes.reserve(256);double prefill=0,decode=0,firstDecode=0;int musicOffset=metadata["music_offset"].get<int>();
    for(int i=0;i<256;++i){trace.gate();auto begin=Clock::now();auto logits=llm->forward(i?std::vector<int>{musicOffset+codes.back()}:prefix,i==0);require(logits.get()!=nullptr&&logits->getInfo()!=nullptr,"LLM returned no logits");auto size=logits->getInfo()->size;require(size>=16385,"Unexpected music logits shape");const float* data=logits->readMap<float>();require(data,"LLM output synchronization failed");data+=size-16385;int best=0;for(int j=0;j<16385;++j)require(std::isfinite(data[j]),"Non-finite music logits");for(int j=1;j<16384;++j)if(data[j]>data[best])best=j;codes.push_back(best);double spent=seconds(begin);if(i==0)prefill=spent;else {decode+=spent;if(i==1)firstDecode=spent;}if(i==0||((i+1)%64)==0)trace.mark("llm-codes-"+std::to_string(i+1));}
    result["prefillSeconds"]=prefill;result["decodeSeconds"]=decode;result["decodeTokensPerSecond"]=255/decode;result["prefillTokens"]=prefix.size();result["generatedCodecIds"]=codes;result["kvPositionsAfterDecode"]=llm->getCurrentHistory();result["eosSuppressedForFixedLengthDiagnostic"]=true;
    result["firstDecodeSeconds"]=firstDecode;result["warmDecodeTokensPerSecond"]=254/(decode-firstDecode);
    trace.mark("llm-256-codes-complete");
    if(llmOnly){llm.reset();trace.mark("llm-only-destroyed");result["scope"]="Same26-layer diagnostic LLM short/long prefix comparison; no DiT/VAE or audio";result.erase("validPcmSeconds");result["success"]=true;result["llmOnly"]=true;return finish();}
    if(policy=="serial"){llm.reset();trace.mark("llm-destroyed-before-dit");}
    if(!dit){trace.gate(1536);dit=std::make_unique<Session>(dir+"/dit/"+ditName,false);trace.mark("dit-loaded");}
    trace.gate();
    auto codebook=read(dir+"/aux/codebook.f32",16384*32),weight=read(dir+"/aux/weight.f32",1024*32),bias=read(dir+"/aux/bias.f32",1024),mask=read(dir+"/aux/mask.f32",3*48),mean=read(dir+"/aux/mean.f32",64),stddev=read(dir+"/aux/std.f32",64);
    std::vector<float> mu(256*1024);for(int t=0;t<256;++t)for(int c=0;c<1024;++c){float sum=bias[c];for(int k=0;k<32;++k)sum+=codebook[codes[t]*32+k]*weight[c*32+k];mu[t*1024+c]=sum;}
    result["conditionerPeak"]=peak(mu);std::mt19937 rng(2307);std::normal_distribution<float> normal;std::vector<float> x(256*64);for(auto& value:x)value=normal(rng);
    double ditTime=0;for(int step=0;step<2;++step){trace.gate();std::vector<float> inputs(2*256*1200,0);for(int b=0;b<2;++b)for(int t=0;t<256;++t){size_t offset=(b*256+t)*1200;std::copy_n(mask.data()+2*48,48,inputs.data()+offset);if(b)std::copy_n(mu.data()+t*1024,1024,inputs.data()+offset+112);std::copy_n(x.data()+t*64,64,inputs.data()+offset+1136);}auto begin=Clock::now();auto velocity=dit->run({{"inputs",std::move(inputs)},{"timestep",{step*.5f,step*.5f}},{"attention_mask",std::vector<float>(2*256*256,1)}},"velocity",2*256*64);ditTime+=seconds(begin);for(size_t j=0;j<x.size();++j)x[j]+=.5f*(velocity[j]+1.5f*(velocity[x.size()+j]-velocity[j]));peak(x);trace.mark("dit-step-"+std::to_string(step+1));}
    result["ditTwoStepSeconds"]=ditTime;result["latentPeak"]=peak(x);
    if(policy!="all"){dit.reset();trace.mark("dit-destroyed-before-vae");}
    if(!vae){trace.gate(1536);vae=std::make_unique<Session>(dir+"/vae/vae-256-fp32.mnn",true);trace.mark("vae-loaded");}
    trace.gate();std::vector<float> latent(64*256);for(int t=0;t<256;++t)for(int c=0;c<64;++c)latent[c*256+t]=x[t*64+c]*stddev[c]+mean[c];auto begin=Clock::now();auto pcm=vae->run({{"latent",std::move(latent)}},"pcm",2*256*1920);result["vaeSeconds"]=seconds(begin);result["pcmPeak"]=peak(pcm);double energy=0;for(float value:pcm)energy+=double(value)*value;result["pcmRms"]=std::sqrt(energy/pcm.size());trace.mark("pcm-finite-10.24s");
    result["pipelineSecondsThroughPcm"]=seconds(trace.start);result["pipelineRtfThroughPcm"]=result["pipelineSecondsThroughPcm"].get<double>()/10.24;result["pipelinePeakRssKiB"]=trace.maxRss.load();
    result["allThreeSessionsRetainedThroughForward"]=(policy=="all");result["llmKvRetainedThroughCodec"]=(policy!="serial");vae.reset();dit.reset();llm.reset();trace.mark("all-models-destroyed");std::this_thread::sleep_for(std::chrono::milliseconds(1000));trace.mark("release-plus-1-second");
    // A separate post-timing session validates the complete phone graph; it
    // must not prewarm the two-step generation whose cold-start RTF is timed.
    trace.gate(1536);dit=std::make_unique<Session>(dir+"/dit/"+ditName,false);
    {auto actual=dit->run({{"inputs",read(dir+"/dit/inputs.f32",2*256*1200)},{"timestep",read(dir+"/dit/timestep.f32",2)},{"attention_mask",read(dir+"/dit/attention_mask.f32",2*256*256)}},"velocity",2*256*64);double error=relative(actual,read(dir+"/dit/"+referenceName,2*256*64));result["phoneVsPcSameQuantVelocityL2"]=error;result["strictNumericalParityPassed"]=error<.01;result["strictNumericalParityThreshold"]=.01;trace.mark("post-timing-full-dit-parity-measured");}
    dit.reset();result["flowAndMemoryPassed"]=true;result["success"]=true;
  }catch(const std::exception& error){result["error"]=error.what();trace.mark("stopped");}
  return finish();
}
static std::string runChunks(const std::string& dir,bool tuned,const std::string& vaeVariant,int chunks,const std::string& ditSubdir,bool overlapLoading,int llmThreads,bool overlapCompute=false){
  constexpr int frames=256,overlapFrames=64,hop=192;
  constexpr size_t pcmWindow=frames*1920,pcmOverlap=overlapFrames*1920;
  Trace trace;trace.sink=dir+"/last-chain-progress.jsonl";{std::ofstream clear(trace.sink);}
  Json result={{"success",false},{"flowAndMemoryPassed",false},{"policy","all-five-chunks"},
    {"scope","Five sequential diagnostic codec windows, one LLM KV and one DiT/VAE session each; RANDOM music adapters; no speaker playback or quality proof"},
    {"music_trained",false},{"real_song_generated",false},{"requestedChunks",chunks},{"windowFrames",frames},
    {"overlapFrames",overlapFrames},{"ditStepsPerChunk",2},{"ditWeightBits",8},{"sampleIntervalMs",20},
    {"oneLlmSession",true},{"onePrefill",true},{"oneDitSession",true},{"oneVaeSession",true},
    {"pcmAccumulatedAcrossWholeSong",false},{"maxCodeWindowIds",frames},{"retainedPcmTailSamplesPerChannel",pcmOverlap}};
  result["policy"]="bounded-chunks";result["ditSubdirectory"]=ditSubdir;result["overlapLoading"]=overlapLoading;result["llmThreads"]=llmThreads;
  result["scope"]="Bounded sequential numerical codec windows; random music adapters and optional untrained smaller DiT; not speaker playback or trained music quality";
  result["overlapLlmAndCodecCompute"]=overlapCompute;result["maximumPendingTokenBatches"]=overlapCompute?1:0;
  result["variant"]=tuned?"dit-low-vae8-llm-highpower":"baseline-high-vae6";result["ditPrecision"]=tuned?"Low":"High";result["vaeThreads"]=tuned?8:6;result["llmPowerHint"]=tuned?"High":"Normal";result["eosSuppressedForFixedLengthDiagnostic"]=true;
  try{
    require(chunks>=1&&chunks<=10,"Bounded experiment supports 1..10 chunks");
    require(ditSubdir=="dit"||ditSubdir=="small-dit","Invalid DiT directory");
    require(llmThreads==4||llmThreads==6,"Invalid LLM thread count");
    require(!(overlapCompute&&overlapLoading),"Test loader and compute overlap independently");
    result["ditArchitectureOnly"]=(ditSubdir=="small-dit");
    require(vaeVariant=="fp32"||vaeVariant=="w8"||vaeVariant=="w8-vsin"||vaeVariant=="w8-snake","Invalid VAE variant");
#ifndef LEVO_VAE_VECTOR_SIN
    require(vaeVariant!="w8-vsin"&&vaeVariant!="w8-snake","Optimized VAE runtime not compiled");
#endif
    require(vaeVariant!="w8-snake"||!tuned,"Snake runtime requires 6-thread baseline config");
    result["vaeVariant"]=vaeVariant;result["vaeWeightBits"]=vaeVariant=="fp32"?32:8;result["vaeVectorSinRequested"]=vaeVariant=="w8-vsin";result["vaeSnakeFusionRequested"]=vaeVariant=="w8-snake";
    trace.gate(2048);trace.mark("five-chunk-start");
    auto metadata=readJson(dir+"/student/diagnostic-student.json");require(metadata["music_trained"]==false&&metadata["layers"]==26,"Expected diagnostic student");
    auto prefix=readJson(dir+"/student/prefix.json").get<std::vector<int>>();require(prefix.size()==514,"Expected514position prefix");
    auto config=readJson(dir+"/student/config.json");config["base_dir"]=dir+"/student/";config["llm_config"]="llm_config.json";config["embedding_file"]="embeddings_bf16.bin";config["reuse_kv"]=false;
    config["thread_num"]=llmThreads;
    if(tuned)config["power"]="high";
    std::string runtime=dir+"/student/config-runtime.json";{std::ofstream out(runtime);out<<config.dump(2);require(out.good(),"Cannot save runtime config");}
    std::unique_ptr<Llm,decltype(&Llm::destroy)> llm(Llm::createLLM(runtime),&Llm::destroy);require(llm&&llm->load(),"LLM load failed");trace.mark("five-llm-loaded");trace.gate();
    std::unique_ptr<Session> dit,vae;
    auto makeDit=[&](){return std::make_unique<Session>(dir+"/"+ditSubdir+"/full-w8.mnn",tuned);};
    auto makeVae=[&](){
#ifdef LEVO_VAE_VECTOR_SIN
    LevoVaeSin::Scope sinScope(vaeVariant=="w8-vsin");
#endif
    return std::make_unique<Session>(dir+"/vae/vae-256-"+(vaeVariant=="fp32"?"fp32":"w8")+".mnn",true,tuned?8:6,vaeVariant=="w8-snake");};
    std::atomic<double> ditLoadSeconds{0},vaeLoadSeconds{0};
    auto loadDit=[&](){auto tick=Clock::now();auto value=makeDit();ditLoadSeconds=seconds(tick);return value;};
    auto loadVae=[&](){auto tick=Clock::now();auto value=makeVae();vaeLoadSeconds=seconds(tick);return value;};
    std::future<std::unique_ptr<Session>> ditFuture,vaeFuture;
    if(!overlapLoading){dit=loadDit();trace.mark("five-dit-loaded");trace.gate();vae=loadVae();trace.mark("five-all-three-loaded");trace.gate();}
    result["initializationSeconds"]=seconds(trace.start);
    auto codebook=read(dir+"/aux/codebook.f32",16384*32),weight=read(dir+"/aux/weight.f32",1024*32),bias=read(dir+"/aux/bias.f32",1024),mask=read(dir+"/aux/mask.f32",3*48),mean=read(dir+"/aux/mean.f32",64),stddev=read(dir+"/aux/std.f32",64);
    llm->reset();llm->generate_init();int generated=0,lastCode=-1,musicOffset=metadata["music_offset"].get<int>();uint64_t codeHash=1469598103934665603ULL;
    std::vector<int> codeTail;std::vector<float> context(overlapFrames*64,0),pcmTail;
    std::mt19937 rng(2307);std::normal_distribution<float> normal;
    size_t emittedSamples=0;double energy=0,maxPcm=0,prefillSeconds=0,decodeSeconds=0,sumDit=0,sumVae=0;
    Json rows=Json::array();
    struct TokenBatch{std::vector<int> fresh;int total=0;size_t kv=0;double llmSeconds=0,started=0,finished=0;};
    auto produce=[&](int newCodes,int chunk){TokenBatch batch;batch.started=seconds(trace.start);batch.fresh.reserve(newCodes);
      for(int i=0;i<newCodes;++i){trace.gate();auto begin=Clock::now();auto logits=llm->forward(generated?std::vector<int>{musicOffset+lastCode}:prefix,generated==0);
        require(logits.get()!=nullptr&&logits->getInfo()!=nullptr,"LLM resume returned no logits");auto size=logits->getInfo()->size;require(size>=16385,"Wrong music logit shape");const float* data=logits->readMap<float>();require(data,"LLM resume synchronization failed");data+=size-16385;int best=0;for(int j=0;j<16385;++j)require(std::isfinite(data[j]),"Non-finite LLM output");for(int j=1;j<16384;++j)if(data[j]>data[best])best=j;
        lastCode=best;batch.fresh.push_back(best);double elapsed=seconds(begin);batch.llmSeconds+=elapsed;if(generated==0)prefillSeconds=elapsed;else decodeSeconds+=elapsed;++generated;
        codeHash=(codeHash^uint64_t(best))*1099511628211ULL;
        if(overlapLoading&&chunk==0&&i==127){trace.mark("dit-load-dispatched-mid-llm");ditFuture=std::async(std::launch::async,loadDit);}
      }
      batch.total=generated;batch.kv=llm->getCurrentHistory();batch.finished=seconds(trace.start);require(batch.kv==prefix.size()+generated-1,"KV was reset or advanced incorrectly across codec pause");return batch;};
    std::future<TokenBatch> pendingCodes;
    for(int chunk=0;chunk<chunks;++chunk){
      trace.gate();auto chunkStart=Clock::now();int newCodes=chunk?hop:frames;auto waitStart=Clock::now();
      auto batch=(overlapCompute&&chunk)?pendingCodes.get():produce(newCodes,chunk);
      double tokenWait=seconds(waitStart);std::vector<int> codes=codeTail;codes.insert(codes.end(),batch.fresh.begin(),batch.fresh.end());
      const int completedCodes=batch.total;const size_t expectedKv=batch.kv;const double llmSeconds=batch.llmSeconds;
      require(codes.size()==frames,"Bounded code window changed");trace.mark("chunk-"+std::to_string(chunk+1)+"-llm-ready");
      if(overlapCompute&&chunk+1<chunks){trace.mark("llm-next-batch-dispatched");pendingCodes=std::async(std::launch::async,[&,next=chunk+1](){return produce(hop,next);});}
      if(!dit){auto tick=Clock::now();dit=ditFuture.get();result["ditLoadJoinWaitSeconds"]=seconds(tick);trace.mark("overlap-dit-ready");trace.gate();}
      std::vector<float> mu(frames*1024);for(int t=0;t<frames;++t)for(int c=0;c<1024;++c){float value=bias[c];for(int k=0;k<32;++k)value+=codebook[codes[t]*32+k]*weight[c*32+k];mu[t*1024+c]=value;}peak(mu);
      std::vector<float> noise(frames*64);for(auto& value:noise)value=normal(rng);std::vector<float> x=noise;double ditSeconds=0;
      for(int step=0;step<2;++step){
        trace.gate();float time=step*.5f;if(chunk)for(int t=0;t<overlapFrames;++t)for(int c=0;c<64;++c)x[t*64+c]=(1-(1-1e-4f)*time)*noise[t*64+c]+time*context[t*64+c];
        std::vector<float> inputs(2*frames*1200,0);
        for(int b=0;b<2;++b)for(int t=0;t<frames;++t){size_t offset=(b*frames+t)*1200;bool known=chunk&&t<overlapFrames;std::copy_n(mask.data()+(known?48:96),48,inputs.data()+offset);if(known)std::copy_n(context.data()+t*64,64,inputs.data()+offset+48);if(b)std::copy_n(mu.data()+t*1024,1024,inputs.data()+offset+112);std::copy_n(x.data()+t*64,64,inputs.data()+offset+1136);}
        auto begin=Clock::now();auto velocity=dit->run({{"inputs",std::move(inputs)},{"timestep",{time,time}},{"attention_mask",std::vector<float>(2*frames*frames,1)}},"velocity",2*frames*64);ditSeconds+=seconds(begin);
        for(size_t j=0;j<x.size();++j)x[j]+=.5f*(velocity[j]+1.5f*(velocity[x.size()+j]-velocity[j]));peak(x);trace.mark("chunk-"+std::to_string(chunk+1)+"-dit-"+std::to_string(step+1));
        if(overlapLoading&&chunk==0&&step==0){trace.mark("vae-load-dispatched-mid-dit");vaeFuture=std::async(std::launch::async,loadVae);}
      }
      if(chunk)std::copy(context.begin(),context.end(),x.begin());
      std::vector<float> latent(64*frames);for(int t=0;t<frames;++t)for(int c=0;c<64;++c)latent[c*frames+t]=x[t*64+c]*stddev[c]+mean[c];
      if(!vae){auto tick=Clock::now();vae=vaeFuture.get();result["vaeLoadJoinWaitSeconds"]=seconds(tick);trace.mark("overlap-vae-ready");}
      trace.gate();auto begin=Clock::now();auto pcm=vae->run({{"latent",std::move(latent)}},"pcm",2*pcmWindow);double vaeSeconds=seconds(begin);maxPcm=std::max(maxPcm,peak(pcm));
      if(chunk){require(pcmTail.size()==2*pcmOverlap,"Missing bounded PCM tail");for(int channel=0;channel<2;++channel)for(size_t i=0;i<pcmOverlap;++i){float alpha=float(i)/float(pcmOverlap-1);pcm[channel*pcmWindow+i]=pcmTail[channel*pcmOverlap+i]*(1-alpha)+pcm[channel*pcmWindow+i]*alpha;}}
      size_t count=chunk==chunks-1?pcmWindow:pcmWindow-pcmOverlap;
      for(int channel=0;channel<2;++channel)for(size_t i=0;i<count;++i){float value=pcm[channel*pcmWindow+i];require(std::isfinite(value),"Non-finite emitted PCM");energy+=double(value)*value;}
      emittedSamples+=count;
      if(chunk<chunks-1){pcmTail.resize(2*pcmOverlap);for(int channel=0;channel<2;++channel)std::copy_n(pcm.data()+channel*pcmWindow+pcmWindow-pcmOverlap,pcmOverlap,pcmTail.data()+channel*pcmOverlap);std::copy_n(x.data()+(frames-overlapFrames)*64,overlapFrames*64,context.data());codeTail.assign(codes.end()-overlapFrames,codes.end());}
      else {pcmTail.clear();codeTail.clear();}
      sumDit+=ditSeconds;sumVae+=vaeSeconds;trace.mark("chunk-"+std::to_string(chunk+1)+"-pcm-complete");
      double wall=seconds(trace.start);if(!chunk)result["firstNumericalPacketSeconds"]=wall;
      rows.push_back({{"chunk",chunk+1},{"newCodecIds",newCodes},{"generatedCodesTotal",completedCodes},{"kvPositions",expectedKv},{"contextFrames",chunk?overlapFrames:0},
        {"llmWorkStartSeconds",batch.started},{"llmWorkEndSeconds",batch.finished},{"tokenBatchWaitSeconds",tokenWait},
        {"llmSeconds",llmSeconds},{"ditTwoStepSeconds",ditSeconds},{"vaeSeconds",vaeSeconds},{"chunkSeconds",seconds(chunkStart)},
        {"emittedPcmSeconds",double(count)/48000},{"cumulativePcmSeconds",double(emittedSamples)/48000},{"wallSeconds",wall},
        {"rssKiB",resident()},{"systemAvailableKiB",available()},{"runningRtf",wall/(double(emittedSamples)/48000)}});
    }
    require(generated==frames+(chunks-1)*hop&&emittedSamples==size_t(generated)*1920,"Dropped or duplicated codec IDs/PCM");
    result["chunks"]=rows;result["generatedCodecIds"]=generated;result["finalKvPositions"]=llm->getCurrentHistory();result["prefillTokens"]=prefix.size();result["prefillSeconds"]=prefillSeconds;result["decodeSeconds"]=decodeSeconds;result["decodeTokensPerSecond"]=(generated-1)/decodeSeconds;result["sumDitSeconds"]=sumDit;result["sumVaeSeconds"]=sumVae;
    result["ditSessionLoadSeconds"]=ditLoadSeconds.load();result["vaeSessionLoadSeconds"]=vaeLoadSeconds.load();
    result["generatedCodeHashFnv64"]=std::to_string(codeHash);
    result["validPcmSeconds"]=double(emittedSamples)/48000;result["pcmRms"]=std::sqrt(energy/(2*emittedSamples));result["pcmPeak"]=maxPcm;result["vaeSnakeFusion"]=vae->snakeStats();
#ifdef LEVO_VAE_VECTOR_SIN
    result["vaeVectorSinKernelCalls"]=LevoVaeSin::calls.load();
#endif
    result["pipelineSecondsThroughPcm"]=seconds(trace.start);result["pipelineRtfThroughPcm"]=result["pipelineSecondsThroughPcm"].get<double>()/result["validPcmSeconds"].get<double>();result["pipelinePeakRssKiB"]=trace.maxRss.load();
    vae.reset();dit.reset();llm.reset();trace.mark("five-all-models-destroyed");std::this_thread::sleep_for(std::chrono::milliseconds(1000));trace.mark("five-release-plus-1-second");
    const auto fixtureDir=dir+"/"+ditSubdir;trace.gate(1536);dit=makeDit();auto actual=dit->run({{"inputs",read(fixtureDir+"/inputs.f32",2*256*1200)},{"timestep",read(fixtureDir+"/timestep.f32",2)},{"attention_mask",read(fixtureDir+"/attention_mask.f32",2*256*256)}},"velocity",2*256*64);
    double error=relative(actual,read(fixtureDir+"/w8-reference.f32",2*256*64));result["phoneVsPcSameQuantVelocityL2"]=error;result["strictNumericalParityPassed"]=error<.01;dit.reset();trace.mark("five-post-timing-parity-measured");result["flowAndMemoryPassed"]=true;result["success"]=true;
  }catch(const std::exception& error){result["error"]=error.what();trace.mark("five-chunk-stopped");}
  trace.sample();result["stages"]=trace.stages;result["sampledPeakRssKiB"]=trace.maxRss.load();result["minimumSystemAvailableKiB"]=trace.minAvailable.load();result["totalSeconds"]=seconds(trace.start);return result.dump();
}
std::string LevoRunFiveChunks(const std::string& dir,bool tuned,const std::string& variant){return runChunks(dir,tuned,variant,5,"dit",false,4);}
std::string LevoRunChunks(const std::string& dir,int chunks,const std::string& ditSubdir,bool overlap,int threads,bool compute){return runChunks(dir,false,"w8-vsin",chunks,ditSubdir,overlap,threads,compute);}
#ifndef LEVO_CHAIN_LIBRARY
int main(int argc,char** argv){if(argc!=3)return 2;std::string policy=argv[2];auto result=policy=="chunks-5"?LevoRunFiveChunks(argv[1]):policy=="chunks-5-w8-vsin"?LevoRunFiveChunks(argv[1],false,"w8-vsin"):policy=="chunks-5-w8-snake"?LevoRunFiveChunks(argv[1],false,"w8-snake"):LevoRunChainBench(argv[1],policy);std::cout<<"LEVO_CHAIN_REPORT "<<result<<std::endl;return Json::parse(result)["success"].get<bool>()?0:1;}
#endif
