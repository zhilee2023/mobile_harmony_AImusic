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
using Json=nlohmann::json;using Clock=std::chrono::steady_clock;
static void require(bool value,const char* error){if(!value)throw std::runtime_error(error);}
static std::vector<float> read(const std::string& name,size_t count){std::ifstream f(name,std::ios::binary|std::ios::ate);require(f&&f.tellg()==std::streamoff(count*4),"Fixture mismatch");std::vector<float> data(count);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),count*4);return data;}
static bool ends(const std::string& s,const std::string& suffix){return s.size()>=suffix.size()&&s.compare(s.size()-suffix.size(),suffix.size(),suffix)==0;}
struct Group{MNN::Tensor* input=nullptr;MNN::Tensor* output=nullptr;MNN::Tensor* alphaTensor=nullptr;MNN::Tensor* betaTensor=nullptr;std::vector<float> alpha,beta;};
static std::vector<float> host(MNN::Tensor* t){MNN::Tensor h(t,MNN::Tensor::CAFFE,true);require(t->copyToHostTensor(&h),"Coefficient read failed");return std::vector<float>(h.host<float>(),h.host<float>()+h.elementSize());}
static void fused(Group& g){
 auto* desc=MNN::TensorUtils::getDescribe(g.input);auto* outputDesc=MNN::TensorUtils::getDescribe(g.output);
 require(desc->dimensionFormat==MNN::MNN_DATA_FORMAT_NCHW&&outputDesc->dimensionFormat==MNN::MNN_DATA_FORMAT_NCHW,"Fusion requires unchanged NCHW layout");
 require(MNN::CPUBackend::getBytes(MNN::TensorUtils::getDescribeOrigin(g.input)->getBackend(),g.input)==2&&MNN::CPUBackend::getBytes(MNN::TensorUtils::getDescribeOrigin(g.output)->getBackend(),g.output)==2,"Fusion requires actual FP16 storage");
 const int channels=g.input->length(1),time=g.input->length(2);require(g.alpha.size()==channels&&g.beta.size()==channels,"Fusion coefficient shape mismatch");
 auto* x=g.input->host<float16_t>();auto* y=g.output->host<float16_t>();require(x&&y,"Missing FP16 storage");
 auto* cpuBackend=static_cast<MNN::CPUBackend*>(MNN::TensorUtils::getDescribeOrigin(g.input)->getBackend());
 require(cpuBackend!=nullptr,"Missing CPU backend for shared thread pool");
 MNN::ThreadPool::TASK task;task.second=6;task.first=[&](int worker){
  for(int c=worker;c<channels;c+=6){auto a=vdupq_n_f16(float16_t(g.alpha[c])),b=vdupq_n_f16(float16_t(g.beta[c]));int i=0;
   for(;i+8<=time;i+=8){auto original=vld1q_f16(x+c*time+i);auto z=vmulq_f16(original,a);float16x8_t s;
    if(vmaxvq_f16(vabsq_f16(z))>128){alignas(16) float16_t tmp[8];vst1q_f16(tmp,z);for(auto& q:tmp)q=float16_t(std::sin(float(q)));s=vld1q_f16(tmp);}
    else s=vcombine_f16(vcvt_f16_f32(LevoVaeSin::sine4(vcvt_f32_f16(vget_low_f16(z)))),vcvt_f16_f32(LevoVaeSin::sine4(vcvt_f32_f16(vget_high_f16(z)))));
    auto square=vmulq_f16(s,s);auto scaled=vmulq_f16(square,b);vst1q_f16(y+c*time+i,vaddq_f16(original,scaled));
   }
   for(;i<time;++i){float16_t z=float16_t(x[c*time+i]*float16_t(g.alpha[c]));float16_t s=float16_t(std::sin(float(z)));float16_t q=float16_t(s*s);q=float16_t(q*float16_t(g.beta[c]));y[c*time+i]=float16_t(x[c*time+i]+q);}
  }
 };cpuBackend->enqueue(task);
}
int main(int argc,char** argv){Json result={{"success",false},{"scope","Callback Snake fusion feasibility only; original graph allocations retained; not whole-chain RTF"}};
 try{require(argc==2,"Usage: levo_vae_snake_fusion fixtureDir");std::string dir=argv[1];
  std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/vae-256-w8.mnn").c_str()));require(bool(net),"Missing graph");net->setExternalFile((dir+"/vae-256-w8.mnn.weight").c_str());net->setSessionMode(MNN::Interpreter::Session_Debug);net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
  MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;
  auto* session=net->createSession(config);require(session,"Session failed");net->releaseModel();std::map<std::string,Group> groups;size_t fusedCalls=0;
  auto warmBefore=[&](const std::vector<MNN::Tensor*>& inputs,const MNN::OperatorInfo* info){std::string name=info->name();if(ends(name,"/Mul_1_output_0")){auto& g=groups[name.substr(0,name.size()-15)];int small=inputs.at(0)->elementSize()<inputs.at(1)->elementSize()?0:1;g.input=inputs.at(1-small);g.alphaTensor=inputs.at(small);}else if(ends(name,"/Mul_2_output_0")){auto& g=groups[name.substr(0,name.size()-15)];g.betaTensor=inputs.at(inputs.at(0)->elementSize()<inputs.at(1)->elementSize()?0:1);}return true;};
  auto warmAfter=[&](const std::vector<MNN::Tensor*>& outputs,const MNN::OperatorInfo* info){std::string name=info->name();if(ends(name,"/Add_1_output_0"))groups[name.substr(0,name.size()-15)].output=outputs.at(0);return true;};
  auto put=[&](int index){auto values=read(dir+"/latent-"+std::to_string(index)+".f32",64*256);auto* input=net->getSessionInput(session,"latent");MNN::Tensor h(input,MNN::Tensor::CAFFE,true);std::memcpy(h.host<float>(),values.data(),values.size()*4);require(input->copyFromHostTensor(&h),"Input copy failed");};
  std::cout<<"LEVO_SNAKE_STAGE warm-capture-start"<<std::endl;
  put(0);require(net->runSessionWithCallBackInfo(session,warmBefore,warmAfter,true)==MNN::NO_ERROR,"Warm capture failed");result["capturedGroups"]=groups.size();require(groups.size()==36,"Expected all 36 Snake groups");
  std::cout<<"LEVO_SNAKE_STAGE warm-capture-complete"<<std::endl;
  Json coefficients;std::ifstream(dir+"/snake-coefficients.json")>>coefficients;
  require(coefficients.at("groups").size()==36,"Expected 36 immutable coefficient groups");
  for(auto& pair:groups){auto& g=pair.second;require(g.input&&g.output,"Incomplete Snake group");const auto& values=coefficients.at("groups").at(pair.first);g.alpha=values.at("alpha").get<std::vector<float>>();g.beta=values.at("beta").get<std::vector<float>>();require(!g.alpha.empty()&&g.alpha.size()<=2048&&g.beta.size()==g.alpha.size(),"Coefficient capture must stay bounded");}
  std::cout<<"LEVO_SNAKE_STAGE immutable-coefficients-ready"<<std::endl;
  auto before=[&](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo* info){std::string name=info->name();auto slash=name.rfind('/');if(slash==std::string::npos)return true;auto found=groups.find(name.substr(0,slash));if(found==groups.end())return true;
   if(ends(name,"/Add_1_output_0")){fused(found->second);++fusedCalls;return false;}
   if(ends(name,"/Mul_1_output_0")||ends(name,"/Mul_2_output_0")||ends(name,"/Sin_output_0")||ends(name,"/Pow_output_0"))return false;return true;};
  double seconds=0;for(int index=0;index<3;++index){std::cout<<"LEVO_SNAKE_STAGE fused-window-"<<index+1<<"-start"<<std::endl;put(index);auto begin=Clock::now();require(net->runSessionWithCallBackInfo(session,before,[](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo*){return true;},true)==MNN::NO_ERROR,"Fused run failed");auto* output=net->getSessionOutput(session,"pcm");auto actual=host(output);double spent=std::chrono::duration<double>(Clock::now()-begin).count();seconds+=spent;auto reference=read(dir+"/runtime-w8-"+std::to_string(index)+".f32",actual.size());double error=0,energy=0;for(size_t i=0;i<actual.size();++i){require(std::isfinite(actual[i]),"Nonfinite fused PCM");double d=actual[i]-reference[i];error+=d*d;energy+=double(reference[i])*reference[i];}double l2=std::sqrt(error/std::max(energy,1e-30));result["rows"].push_back({{"seconds",spent},{"l2VsOriginalPhoneW8",l2}});require(l2<.001,"Fusion numeric equivalence failed");}
  result["computeSeconds"]=seconds;result["vaeOnlyRtf"]=seconds/25.6;result["fusedCalls"]=fusedCalls;result["success"]=true;net->releaseSession(session);
 }catch(const std::exception& e){result["error"]=e.what();}std::cout<<"LEVO_SNAKE_FUSION "<<result.dump()<<std::endl;return result["success"].get<bool>()?0:1;}
