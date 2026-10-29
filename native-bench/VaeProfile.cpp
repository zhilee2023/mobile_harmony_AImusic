#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN_generated.h>
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
#include <string>
#include <vector>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
struct Entry{double seconds=0,mflops=0;int calls=0;std::string type;Json inputs,outputs;};
int main(int argc,char** argv){
  Json report={{"success",false},{"scope","VAE-only CPU operator callback profile, one warm run then one profiled run; not whole-codec RTF"},
    {"threadsRequested",6},{"powerHint","High"},{"memoryMode","Low"},{"kleidiaiEnabled",false}};
  try{
    if(argc<3||argc>4)throw std::runtime_error("Usage: levo_vae_profile fixtureDir high|low [fp32|w8]");
    std::string dir=argv[1],precision=argv[2];if(precision!="high"&&precision!="low")throw std::runtime_error("Invalid precision");
    std::string variant=argc>3?argv[3]:"fp32";if(variant!="fp32"&&variant!="w8")throw std::runtime_error("Invalid weight variant");report["weightVariant"]=variant;
    Json manifest;std::ifstream file(dir+"/fixture.json");file>>manifest;int frames=manifest.at("frames").get<int>();
    if(frames!=64&&frames!=256)throw std::runtime_error("Unsupported bounded profile shape");report["frames"]=frames;
    std::vector<float> latent(64*frames);std::ifstream inputFile(dir+"/latent-0.f32",std::ios::binary|std::ios::ate);
    if(!inputFile||inputFile.tellg()!=std::streamoff(latent.size()*4))throw std::runtime_error("Bad profile input");
    inputFile.seekg(0);inputFile.read(reinterpret_cast<char*>(latent.data()),latent.size()*4);
    std::string model=dir+"/vae-"+std::to_string(frames)+"-"+variant+".mnn";
    std::ifstream graphFile(model,std::ios::binary|std::ios::ate);if(!graphFile)throw std::runtime_error("Missing graph metadata");
    std::vector<char> graphBytes(size_t(graphFile.tellg()));graphFile.seekg(0);graphFile.read(graphBytes.data(),graphBytes.size());
    flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(graphBytes.data()),graphBytes.size());
    if(!MNN::VerifyNetBuffer(verifier))throw std::runtime_error("Invalid graph metadata");
    const auto* graph=MNN::GetNet(graphBytes.data());Json convolutions=Json::array();
    for(const auto* op:*graph->oplists()){if(op->main_type()!=MNN::OpParameter_Convolution2D)continue;
      const auto* conv=op->main_as_Convolution2D();const auto* common=conv->common();const auto* quant=conv->quanParameter();
      Json row={{"name",op->name()->str()},{"type",MNN::EnumNameOpType(op->type())},{"inputChannels",common->inputCount()},{"outputChannels",common->outputCount()},
        {"kernelX",common->kernelX()},{"kernelY",common->kernelY()},{"strideX",common->strideX()},{"strideY",common->strideY()},
        {"dilateX",common->dilateX()},{"dilateY",common->dilateY()},{"hasWeightQuantMetadata",quant!=nullptr}};
      if(quant){row["quantType"]=quant->type();row["aMaxOrBits"]=quant->aMaxOrBits();row["aMin"]=quant->aMin();row["readType"]=quant->readType();}
      convolutions.push_back(row);
    }report["convolutionWeightMetadata"]=convolutions;
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model.c_str()));
    if(!net)throw std::runtime_error("Cannot load decoder");std::string external=model+".weight";net->setExternalFile(external.c_str());
    net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);net->setSessionMode(MNN::Interpreter::Session_Debug);net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
    MNN::BackendConfig backend;backend.precision=precision=="low"?MNN::BackendConfig::Precision_Low:MNN::BackendConfig::Precision_High;
    backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
    MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;
    auto* session=net->createSession(config);if(!session)throw std::runtime_error("Session failed");net->resizeSession(session);net->releaseModel();
    int backends[2]={-1,-1};if(net->getSessionInfo(session,MNN::Interpreter::BACKENDS,backends))report["mainPipelineBackend"]=backends[0];
    auto put=[&](){auto* input=net->getSessionInput(session,"latent");if(!input)throw std::runtime_error("Missing input");MNN::Tensor host(input,MNN::Tensor::CAFFE,true);if(host.elementSize()!=latent.size())throw std::runtime_error("Input shape mismatch");std::memcpy(host.host<float>(),latent.data(),latent.size()*4);if(!input->copyFromHostTensor(&host))throw std::runtime_error("Input copy failed");};
    put();auto begin=Clock::now();if(net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("Warm run failed");
    report["warmRunSeconds"]=std::chrono::duration<double>(Clock::now()-begin).count();
    std::map<std::string,Entry> nodes,types;Clock::time_point opStart;
    auto shapes=[](const std::vector<MNN::Tensor*>& tensors){Json value=Json::array();for(auto* tensor:tensors)value.push_back(tensor->shape());return value;};
    auto before=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){nodes[info->name()].inputs=shapes(tensors);opStart=Clock::now();return true;};
    auto after=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){double dt=std::chrono::duration<double>(Clock::now()-opStart).count();
      auto& node=nodes[info->name()];node.type=info->type();node.seconds+=dt;node.mflops+=info->flops();node.calls++;
      node.outputs=shapes(tensors);
      auto& type=types[info->type()];type.seconds+=dt;type.mflops+=info->flops();type.calls++;return true;};
    put();begin=Clock::now();if(net->runSessionWithCallBackInfo(session,before,after,true)!=MNN::NO_ERROR)throw std::runtime_error("Profiled run failed");
    report["profiledRunSeconds"]=std::chrono::duration<double>(Clock::now()-begin).count();
    auto* output=net->getSessionOutput(session,"pcm");if(!output)throw std::runtime_error("Missing PCM");MNN::Tensor host(output,MNN::Tensor::CAFFE,true);
    if(host.elementSize()!=2*frames*1920||!output->copyToHostTensor(&host))throw std::runtime_error("Bad PCM output");
    for(int i=0;i<host.elementSize();++i)if(!std::isfinite(host.host<float>()[i]))throw std::runtime_error("Non-finite output");
    if(types.empty())throw std::runtime_error("No operator callback evidence collected");
    double sum=0;for(auto& pair:types)sum+=pair.second.seconds;report["callbackSeconds"]=sum;
    Json typeRows=Json::array();for(auto& pair:types)typeRows.push_back({{"type",pair.first},{"seconds",pair.second.seconds},{"calls",pair.second.calls},{"percent",100*pair.second.seconds/std::max(sum,1e-30)},{"mnnEstimatedMflops",pair.second.mflops}});
    std::sort(typeRows.begin(),typeRows.end(),[](const Json& a,const Json& b){return a["seconds"].get<double>()>b["seconds"].get<double>();});
    report["byOperatorType"]=typeRows;std::vector<std::pair<std::string,Entry>> sorted(nodes.begin(),nodes.end());
    std::sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second.seconds>b.second.seconds;});
    Json top=Json::array();for(size_t i=0;i<std::min(size_t(40),sorted.size());++i){auto& row=sorted[i];top.push_back({{"name",row.first},{"type",row.second.type},{"seconds",row.second.seconds},{"percent",100*row.second.seconds/std::max(sum,1e-30)},{"inputShapes",row.second.inputs},{"outputShapes",row.second.outputs}});}report["topOperators"]=top;
    report["precision"]=precision;report["success"]=true;report["estimatedFlopsAreHardwareCounters"]=false;
    net->releaseSession(session);net.reset();
  }catch(const std::exception& error){report["error"]=error.what();}
  std::cout<<"LEVO_VAE_PROFILE "<<report.dump()<<std::endl;return report["success"].get<bool>()?0:1;
}
