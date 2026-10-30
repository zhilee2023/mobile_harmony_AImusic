// Bounded callback-fusion feasibility probe. Keeps original graph allocations;
// this is not a production graph rewrite or a claim of reduced peak workspace.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "VaeVectorSin.hpp"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <unistd.h>
#include <climits>
using Json=nlohmann::json;using Clock=std::chrono::steady_clock;
static long rss(){long total=0,resident=0;std::ifstream("/proc/self/statm")>>total>>resident;return resident*sysconf(_SC_PAGESIZE)/1024;}
struct RssSamples{std::atomic<bool> stop{false};std::atomic<long> peak{0};std::thread worker;
 RssSamples(){worker=std::thread([this]{while(!stop){long value=rss(),old=peak.load();while(value>old&&!peak.compare_exchange_weak(old,value)){}std::this_thread::sleep_for(std::chrono::milliseconds(20));}});}
 ~RssSamples(){stop=true;worker.join();}};
static void require(bool value,const char* error){if(!value)throw std::runtime_error(error);}
static std::vector<float> read(const std::string& name,size_t count){std::ifstream f(name,std::ios::binary|std::ios::ate);require(f&&f.tellg()==std::streamoff(count*4),"Fixture mismatch");std::vector<float> data(count);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),count*4);return data;}
static bool ends(const std::string& s,const std::string& suffix){return s.size()>=suffix.size()&&s.compare(s.size()-suffix.size(),suffix.size(),suffix)==0;}
// Never unwind through MNN's callbacks: Pipeline::_exitExecute must run first.
static void runSafe(MNN::Interpreter* net,const MNN::Session* session,
                    const MNN::TensorCallBackWithInfo& before,const MNN::TensorCallBackWithInfo& after){
 std::string error;
 auto safeBefore=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){try{return before(tensors,info);}catch(const std::exception& e){error=e.what();return false;}};
 auto safeAfter=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){if(!error.empty())return false;try{return after(tensors,info);}catch(const std::exception& e){error=e.what();return false;}};
 auto code=net->runSessionWithCallBackInfo(session,safeBefore,safeAfter,true);
 require(error.empty(),error.c_str());require(code==MNN::NO_ERROR,"MNN callback forward failed");
}
struct Group{MNN::Tensor* input=nullptr;MNN::Tensor* output=nullptr;MNN::Tensor* alphaTensor=nullptr;MNN::Tensor* betaTensor=nullptr;int originalAddInput=-1;std::string producer;std::vector<float> alpha,beta;};
static std::vector<float> host(MNN::Tensor* t){MNN::Tensor h(t,MNN::Tensor::CAFFE,true);require(t->copyToHostTensor(&h),"Coefficient read failed");return std::vector<float>(h.host<float>(),h.host<float>()+h.elementSize());}
static MNN::Tensor* storage(MNN::Tensor* tensor){
 for(int depth=0;depth<4;++depth){auto* description=MNN::TensorUtils::getDescribe(tensor);
  if(description->memoryType!=MNN::Tensor::InsideDescribe::MEMORY_VIRTUAL)return tensor;
  require(description->regions.size()==1,"Fusion only accepts one full storage region");
  const auto& r=description->regions[0];long elements=long(r.size[0])*r.size[1]*r.size[2];
  require(r.origin&&r.src.offset==0&&r.dst.offset==0&&MNN::TensorUtils::isCopyRegion(r),"Fusion cannot reinterpret a sliced, broadcast or transposed view");
  require(elements==tensor->elementSize()&&elements==r.origin->elementSize(),"Fusion requires a complete reshape view");
  require(tensor->length(1)==r.origin->length(1),"Fusion reshape must preserve channels");tensor=r.origin;
 }throw std::runtime_error("Fusion view chain exceeds bound");
}
static float16x8_t snake8(float16x8_t original,float16x8_t a,float16x8_t b){
 auto z=vmulq_f16(original,a);float16x8_t sine;
 if(vmaxvq_f16(vabsq_f16(z))>128){alignas(16) float16_t values[8];vst1q_f16(values,z);for(auto& q:values)q=float16_t(std::sin(float(q)));sine=vld1q_f16(values);}
 else sine=vcombine_f16(vcvt_f16_f32(LevoVaeSin::sine4(vcvt_f32_f16(vget_low_f16(z)))),vcvt_f16_f32(LevoVaeSin::sine4(vcvt_f32_f16(vget_high_f16(z)))));
 auto square=vmulq_f16(sine,sine);auto scaled=vmulq_f16(square,b);return vaddq_f16(original,scaled);
}
static void fused(Group& g,bool sharedPool,const std::vector<float16_t>& snapshot,MNN::MNN_DATA_FORMAT inFormat){
 auto* outputStorage=storage(g.output);auto* outputBackend=MNN::TensorUtils::getDescribeOrigin(outputStorage)->getBackend();
 require(outputBackend,"Fusion requires live output backend");require(MNN::CPUBackend::getBytes(outputBackend,outputStorage)==2,"Fusion requires actual FP16 storage");
 auto* cpuBackend=static_cast<MNN::CPUBackend*>(outputBackend);auto* outputCpu=cpuBackend;
 const int channels=g.input->length(1),time=g.input->length(2);
 require(g.input->elementSize()==channels*time&&g.output->elementSize()==channels*time,"Only one-batch channel/time tensors supported");
 require(g.alpha.size()==channels&&g.beta.size()==channels,"Fusion coefficient shape mismatch");
 const auto outFormat=MNN::TensorUtils::getDescribe(outputStorage)->dimensionFormat;
 const bool packed=inFormat==MNN::MNN_DATA_FORMAT_NC4HW4&&outFormat==MNN::MNN_DATA_FORMAT_NC4HW4;
 require(packed||(inFormat==MNN::MNN_DATA_FORMAT_NCHW&&outFormat==MNN::MNN_DATA_FORMAT_NCHW),"Mixed layouts require an explicit conversion kernel");
 if(packed){std::cout<<"LEVO_SNAKE_PACK input="<<cpuBackend->functions()->pack<<" output="<<outputCpu->functions()->pack<<std::endl;require(cpuBackend->functions()->bytes==2&&outputCpu->functions()->bytes==2,"CoreFunctions ABI/storage size mismatch");require(cpuBackend->functions()->pack==8&&outputCpu->functions()->pack==8&&channels%8==0,"Packed fusion requires C8 FP16 storage");}
 require(snapshot.size()==size_t(channels)*time,"Producer snapshot size mismatch");const auto* x=snapshot.data();auto* y=outputStorage->host<float16_t>();require(y,"Missing live output storage");
 std::cout<<"LEVO_SNAKE_SAMPLE x0="<<float(x[0])<<" x8="<<float(x[8])<<" alpha0="<<g.alpha[0]<<" inverseBeta0="<<g.beta[0]<<" producerSnapshot=1"<<std::endl;
 std::cout<<"LEVO_SNAKE_STORAGE format="<<(packed?"C8":"NCHW")<<" channels="<<channels<<" time="<<time<<std::endl;
 MNN::ThreadPool::TASK task;task.second=6;task.first=[&](int worker){
  if(packed){for(int c=worker*8;c<channels;c+=6*8){alignas(16) float16_t alpha[8],beta[8];for(int k=0;k<8;++k){alpha[k]=float16_t(g.alpha[c+k]);beta[k]=float16_t(g.beta[c+k]);}
   auto av=vld1q_f16(alpha),bv=vld1q_f16(beta);size_t base=size_t(c/8)*time*8;
   for(int t=0;t<time;++t){auto original=vld1q_f16(x+base+size_t(t)*8);vst1q_f16(y+base+size_t(t)*8,snake8(original,av,bv));}
  }}else{for(int c=worker;c<channels;c+=6){auto av=vdupq_n_f16(float16_t(g.alpha[c])),bv=vdupq_n_f16(float16_t(g.beta[c]));int t=0;
   for(;t+8<=time;t+=8){auto original=vld1q_f16(x+size_t(c)*time+t);vst1q_f16(y+size_t(c)*time+t,snake8(original,av,bv));}
   for(;t<time;++t){float16_t z=float16_t(x[c*time+t]*float16_t(g.alpha[c]));float16_t sine=float16_t(std::sin(float(z)));float16_t square=float16_t(sine*sine),scaled=float16_t(square*float16_t(g.beta[c]));y[c*time+t]=float16_t(x[c*time+t]+scaled);}
  }}
 };
 if(sharedPool)cpuBackend->enqueue(task);else for(int worker=0;worker<6;++worker)task.first(worker);
}
int main(int argc,char** argv){RssSamples samples;Json result={{"success",false},{"scope","Callback Snake fusion feasibility only; original graph allocations retained; not whole-chain RTF"}};
 try{require(argc>=2&&argc<=6,"Usage: levo_vae_snake_fusion fixtureDir [direct|pool] [first|all] [1|3] [neon|scalar]");std::string dir=argv[1];
  std::string dispatch=argc>2?argv[2]:"pool",selection=argc>3?argv[3]:"all";int windows=argc>4?std::stoi(argv[4]):3;
  require(dispatch=="direct"||dispatch=="pool","Invalid dispatch");require(selection=="first"||selection=="all","Invalid selection");require(windows==1||windows==3,"Invalid bounded window count");
  std::string sine=argc>5?argv[5]:"neon";require(sine=="neon"||sine=="scalar","Invalid baseline Sin");
  result["dispatch"]=dispatch;result["selection"]=selection;result["windows"]=windows;result["baselineSin"]=sine;result["pairOrder"]="even windows baseline then fused; odd windows fused then baseline";
  std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/vae-256-w8.mnn").c_str()));require(bool(net),"Missing graph");net->setExternalFile((dir+"/vae-256-w8.mnn.weight").c_str());net->setSessionMode(MNN::Interpreter::Session_Debug);net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
  MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;
  auto* session=net->createSession(config);require(session,"Session failed");LevoVaeSin::Scope sinScope(sine=="neon");net->resizeSession(session);sinScope.restore();net->releaseModel();std::map<std::string,Group> groups;size_t fusedCalls=0;
  std::ifstream modelFile(dir+"/vae-256-w8.mnn",std::ios::binary|std::ios::ate);require(modelFile.good(),"Missing graph metadata");std::vector<char> modelData(size_t(modelFile.tellg()));modelFile.seekg(0);modelFile.read(modelData.data(),modelData.size());
  const auto* model=MNN::GetNet(modelData.data());
  for(const auto* op:*model->oplists()){std::string name=op->name()->str();if(!ends(name,"/Add_1_output_0"))continue;require(op->inputIndexes()&&op->inputIndexes()->size()==2,"Unexpected Add input count");
   auto& group=groups[name.substr(0,name.size()-15)];for(int i=0;i<2;++i){auto inputName=model->tensorName()->Get(op->inputIndexes()->Get(i))->str();if(groups.size()==1)std::cout<<"LEVO_SNAKE_GRAPH_INPUT index="<<i<<" tensor="<<inputName<<std::endl;if(ends(inputName,"/Mul_2_output_0"))group.originalAddInput=1-i;}
   require(group.originalAddInput>=0,"Cannot identify original residual input from graph names");int sourceIndex=op->inputIndexes()->Get(group.originalAddInput);
   for(int depth=0;depth<8;++depth){const MNN::Op* matched=nullptr;
    for(const auto* producer:*model->oplists()){if(!producer->outputIndexes())continue;for(auto outputIndex:*producer->outputIndexes())if(outputIndex==sourceIndex)matched=producer;}
    require(matched,"Original input producer missing");
    if(matched->type()==MNN::OpType_Squeeze||matched->type()==MNN::OpType_Unsqueeze||matched->type()==MNN::OpType_Reshape){require(matched->inputIndexes()&&matched->inputIndexes()->size()>=1,"Invalid reshape producer");sourceIndex=matched->inputIndexes()->Get(0);continue;}
    group.producer=matched->name()->str();break;
   }
   require(!group.producer.empty(),"Original input producer not found");if(groups.size()==1)std::cout<<"LEVO_SNAKE_PRODUCER "<<group.producer<<std::endl;}
  auto warmBefore=[&](const std::vector<MNN::Tensor*>& inputs,const MNN::OperatorInfo* info){std::string name=info->name();if(ends(name,"/Add_1_output_0")){auto& group=groups.at(name.substr(0,name.size()-15));group.input=inputs.at(group.originalAddInput);}return true;};
  auto warmAfter=[&](const std::vector<MNN::Tensor*>& outputs,const MNN::OperatorInfo* info){std::string name=info->name();if(ends(name,"/Add_1_output_0"))groups[name.substr(0,name.size()-15)].output=outputs.at(0);return true;};
  auto put=[&](int index){std::cout<<"LEVO_SNAKE_STAGE input-"<<index<<"-read-start"<<std::endl;auto values=read(dir+"/latent-"+std::to_string(index)+".f32",64*256);auto* input=net->getSessionInput(session,"latent");MNN::Tensor h(input,MNN::Tensor::CAFFE,true);std::memcpy(h.host<float>(),values.data(),values.size()*4);std::cout<<"LEVO_SNAKE_STAGE input-"<<index<<"-copy-start"<<std::endl;require(input->copyFromHostTensor(&h),"Input copy failed");std::cout<<"LEVO_SNAKE_STAGE input-"<<index<<"-ready"<<std::endl;};
  std::cout<<"LEVO_SNAKE_STAGE warm-capture-start"<<std::endl;
  put(0);runSafe(net.get(),session,warmBefore,warmAfter);result["capturedGroups"]=groups.size();require(groups.size()==36,"Expected all 36 Snake groups");
  std::cout<<"LEVO_SNAKE_STAGE warm-capture-complete"<<std::endl;
  {auto actual=host(net->getSessionOutput(session,"pcm"));auto expected=read(dir+"/runtime-w8-0.f32",actual.size());double error=0,energy=0;for(size_t i=0;i<actual.size();++i){double d=actual[i]-expected[i];error+=d*d;energy+=double(expected[i])*expected[i];}double l2=std::sqrt(error/std::max(energy,1e-30));result["warmOriginalL2VsPhoneReference"]=l2;std::cout<<"LEVO_SNAKE_BASELINE l2="<<l2<<std::endl;require(l2<.001,"Unmodified baseline differs from reference; do not attribute error to fusion");}
  Json coefficients;std::ifstream(dir+"/snake-coefficients.json")>>coefficients;
  require(coefficients.at("groups").size()==36,"Expected 36 immutable coefficient groups");
  for(auto& pair:groups){auto& g=pair.second;require(g.input&&g.output,"Incomplete Snake group");const auto& values=coefficients.at("groups").at(pair.first);g.alpha=values.at("alpha").get<std::vector<float>>();g.beta=values.at("beta").get<std::vector<float>>();require(!g.alpha.empty()&&g.alpha.size()<=2048&&g.beta.size()==g.alpha.size(),"Coefficient capture must stay bounded");}
  std::cout<<"LEVO_SNAKE_STAGE immutable-coefficients-ready"<<std::endl;
  std::vector<float16_t> snapshot;std::string snapshotGroup;MNN::MNN_DATA_FORMAT snapshotFormat=MNN::MNN_DATA_FORMAT_NCHW;size_t snapshotPeak=0;
  auto before=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){std::string name=info->name();auto slash=name.rfind('/');if(slash==std::string::npos)return true;auto found=groups.find(name.substr(0,slash));if(found==groups.end()||(selection=="first"&&found!=groups.begin()))return true;
   if(ends(name,"/Add_1_output_0")){found->second.input=tensors.at(found->second.originalAddInput);std::cout<<"LEVO_SNAKE_KERNEL enter "<<found->first<<" shape="<<found->second.input->length(1)<<"x"<<found->second.input->length(2)<<" dispatch="<<dispatch<<std::endl;auto tick=Clock::now();require(snapshotGroup==found->first,"Missing current producer snapshot");fused(found->second,dispatch=="pool",snapshot,snapshotFormat);snapshotGroup.clear();snapshot.clear();++fusedCalls;result["fusedCallsCompleted"]=fusedCalls;std::cout<<"LEVO_SNAKE_KERNEL exit "<<found->first<<" seconds="<<std::chrono::duration<double>(Clock::now()-tick).count()<<std::endl;return false;}
   if(ends(name,"/Mul_1_output_0")||ends(name,"/Mul_2_output_0")||ends(name,"/Sin_output_0")||ends(name,"/Pow_output_0"))return false;return true;};
  auto captureAfter=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){
   for(auto it=groups.begin();it!=groups.end();++it){if(selection=="first"&&it!=groups.begin())continue;if(it->second.producer!=info->name())continue;
    require(snapshotGroup.empty(),"Overlapping producer snapshots need an explicit schedule");auto* source=storage(tensors.at(0));auto* backend=MNN::TensorUtils::getDescribeOrigin(source)->getBackend();require(backend&&MNN::CPUBackend::getBytes(backend,source)==2,"Snapshot requires FP16 producer");
    const size_t count=source->elementSize();require(count<=size_t(128)*491520,"Producer snapshot memory bound exceeded");require(source->length(1)==it->second.alpha.size(),"Producer channels mismatch");snapshotFormat=MNN::TensorUtils::getDescribe(source)->dimensionFormat;
    require(source->host<float16_t>()!=nullptr,"Producer not materialized");snapshot.resize(count);std::memcpy(snapshot.data(),source->host<float16_t>(),count*2);snapshotGroup=it->first;snapshotPeak=std::max(snapshotPeak,count*2);
    std::cout<<"LEVO_SNAKE_CAPTURE "<<it->first<<" bytes="<<count*2<<std::endl;
   }return true;};
  auto relative=[&](const std::vector<float>& actual,int index){auto reference=read(dir+"/runtime-w8-"+std::to_string(index)+".f32",actual.size());double error=0,energy=0;for(size_t i=0;i<actual.size();++i){require(std::isfinite(actual[i]),"Nonfinite PCM");double d=actual[i]-reference[i];error+=d*d;energy+=double(reference[i])*reference[i];}return std::sqrt(error/std::max(energy,1e-30));};
  auto baseline=[&](int index){put(index);auto begin=Clock::now();runSafe(net.get(),session,warmBefore,warmAfter);auto actual=host(net->getSessionOutput(session,"pcm"));double spent=std::chrono::duration<double>(Clock::now()-begin).count();double l2=relative(actual,index);require(l2<.001,"Paired original baseline differs from reference");std::cout<<"LEVO_SNAKE_BASELINE_WINDOW "<<index+1<<" seconds="<<spent<<" l2="<<l2<<std::endl;return spent;};
  double seconds=0,baselineSeconds=0;for(int index=0;index<windows;++index){double base=0;if(index%2==0)base=baseline(index);
   std::cout<<"LEVO_SNAKE_STAGE fused-window-"<<index+1<<"-start"<<std::endl;put(index);std::cout<<"LEVO_SNAKE_STAGE callback-window-"<<index+1<<"-start"<<std::endl;
   auto begin=Clock::now();runSafe(net.get(),session,before,captureAfter);std::cout<<"LEVO_SNAKE_STAGE callback-window-"<<index+1<<"-complete"<<std::endl;
   auto actual=host(net->getSessionOutput(session,"pcm"));double spent=std::chrono::duration<double>(Clock::now()-begin).count();seconds+=spent;
   double l2=relative(actual,index);require(l2<.001,"Fusion numeric equivalence failed");if(index%2)base=baseline(index);baselineSeconds+=base;
   result["rows"].push_back({{"seconds",spent},{"baselineSeconds",base},{"l2VsOriginalPhoneW8",l2}});
  }
  result["baselineComputeSeconds"]=baselineSeconds;result["baselineVaeOnlyRtf"]=baselineSeconds/(windows==3?25.6:10.24);result["speedupVsPairedBaseline"]=baselineSeconds/seconds;result["producerSnapshotPeakBytes"]=snapshotPeak;result["computeSeconds"]=seconds;if(windows==3){result["vaeOnlyRtf"]=seconds/25.6;result["validPcmSeconds"]=25.6;}else{result["singleWindowSeconds"]=10.24;result["vaeOnlyRtf"]=seconds/10.24;}
  result["fusedCalls"]=fusedCalls;result["success"]=true;net->releaseSession(session);
 }catch(const std::exception& e){result["error"]=e.what();}result["vectorSinKernelCalls"]=LevoVaeSin::calls.load();result["sampledPeakRssKiB"]=samples.peak.load();result["sampleIntervalMs"]=20;std::cout<<"LEVO_SNAKE_FUSION "<<result.dump()<<std::endl;return result["success"].get<bool>()?0:1;}
