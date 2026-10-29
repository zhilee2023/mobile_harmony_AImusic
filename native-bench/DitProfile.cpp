#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
using Json=nlohmann::json;using Clock=std::chrono::steady_clock;
struct Entry{double seconds=0;int calls=0;std::string type;};
static std::vector<float> read(const std::string& path,size_t count){std::ifstream in(path,std::ios::binary|std::ios::ate);if(!in||in.tellg()!=std::streamoff(count*4))throw std::runtime_error("Bad profile fixture");std::vector<float> result(count);in.seekg(0);in.read(reinterpret_cast<char*>(result.data()),count*4);return result;}
static std::string category(const std::string& name,const std::string& type){
  if(type=="Convolution"){
    if(name.find("/mlp/c_fc/")!=std::string::npos)return "FFN up projection";
    if(name.find("/mlp/c_proj/")!=std::string::npos)return "FFN down projection";
    if(name.find("/attn/c_attn/")!=std::string::npos)return "QKV projection";
    if(name.find("/attn/c_proj/")!=std::string::npos)return "Attention output projection";
    return "Other linear/convolution";
  }return type;
}
int main(int argc,char** argv){Json result={{"success",false},{"scope","Complete39-layer DiT one warmed velocity evaluation, CLI CPU High6threads, callback/debug overhead; not whole-chain RTF"},{"kleidiaiEnabled",false},{"threads",6},{"memory","Low"},{"power","High"}};
try{
  if(argc!=3)throw std::runtime_error("Usage: levo_dit_profile diagnosticDitDir w8|w4");std::string dir=argv[1],variant=argv[2];if(variant!="w8"&&variant!="w4")throw std::runtime_error("Invalid variant");result["variant"]=variant;
  std::string name=dir+"/full-"+variant+".mnn",weight=name+".weight";std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(name.c_str()));if(!net)throw std::runtime_error("Cannot load complete DiT");net->setExternalFile(weight.c_str());net->setSessionMode(MNN::Interpreter::Session_Debug);net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
  MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_High;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;auto* session=net->createSession(config);if(!session)throw std::runtime_error("Cannot create profile session");net->resizeSession(session);net->releaseModel();
  std::map<std::string,std::vector<float>> values={{"inputs",read(dir+"/inputs.f32",2*256*1200)},{"timestep",read(dir+"/timestep.f32",2)},{"attention_mask",read(dir+"/attention_mask.f32",2*256*256)}};
  auto put=[&](){for(auto& pair:values){auto* input=net->getSessionInput(session,pair.first.c_str());if(!input)throw std::runtime_error("Missing input");MNN::Tensor host(input,MNN::Tensor::CAFFE,true);if(host.elementSize()!=pair.second.size())throw std::runtime_error("Bad input shape");std::memcpy(host.host<float>(),pair.second.data(),pair.second.size()*4);if(!input->copyFromHostTensor(&host))throw std::runtime_error("Input copy failed");}};
  put();auto tick=Clock::now();if(net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("Warm forward failed");result["warmSeconds"]=std::chrono::duration<double>(Clock::now()-tick).count();
  std::map<std::string,Entry> nodes,types,categories;Clock::time_point begin;
  auto before=[&](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo*){begin=Clock::now();return true;};
  auto after=[&](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo* info){double time=std::chrono::duration<double>(Clock::now()-begin).count();for(auto* item:{&nodes[info->name()],&types[info->type()],&categories[category(info->name(),info->type())]}){item->seconds+=time;++item->calls;item->type=info->type();}return true;};
  put();tick=Clock::now();if(net->runSessionWithCallBackInfo(session,before,after,true)!=MNN::NO_ERROR)throw std::runtime_error("Profile forward failed");result["profiledSeconds"]=std::chrono::duration<double>(Clock::now()-tick).count();
  auto* output=net->getSessionOutput(session,"velocity");MNN::Tensor host(output,MNN::Tensor::CAFFE,true);if(host.elementSize()!=2*256*64||!output->copyToHostTensor(&host))throw std::runtime_error("Bad velocity shape");auto reference=read(dir+"/"+variant+"-reference.f32",2*256*64);double error=0,energy=0;for(size_t i=0;i<reference.size();++i){float value=host.host<float>()[i];if(!std::isfinite(value))throw std::runtime_error("Non-finite velocity");double delta=value-reference[i];error+=delta*delta;energy+=double(reference[i])*reference[i];}result["relativeL2VsPcSameQuant"]=std::sqrt(error/std::max(energy,1e-30));
  double total=0;for(auto& pair:types)total+=pair.second.seconds;if(total<=0)throw std::runtime_error("No callback evidence");result["callbackSeconds"]=total;
  auto rows=[&](const std::map<std::string,Entry>& data,int limit){Json output=Json::array();for(auto& pair:data)output.push_back({{"name",pair.first},{"type",pair.second.type},{"seconds",pair.second.seconds},{"calls",pair.second.calls},{"callbackPercent",100*pair.second.seconds/total}});std::sort(output.begin(),output.end(),[](const Json&a,const Json&b){return a["seconds"].get<double>()>b["seconds"].get<double>();});while(output.size()>size_t(limit))output.erase(output.end()-1);return output;};
  result["byType"]=rows(types,100);result["byCategory"]=rows(categories,100);result["topNodes"]=rows(nodes,20);result["success"]=true;net->releaseSession(session);
}catch(const std::exception& error){result["error"]=error.what();}
std::cout<<"LEVO_DIT_PROFILE "<<result.dump()<<std::endl;return result["success"].get<bool>()?0:1;}
