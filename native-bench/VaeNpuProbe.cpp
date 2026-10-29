#include <CANNKit/hiai_single_op.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <json.hpp>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
static void check(bool good,const char* reason){if(!good)throw std::runtime_error(reason);}
static std::vector<float> read(const std::string& file){std::ifstream f(file,std::ios::binary|std::ios::ate);check(f.good(),"Missing case data");size_t bytes=f.tellg();check(bytes%4==0,"Invalid float data");std::vector<float> data(bytes/4);f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),bytes);return data;}
struct Api {
 void* library=nullptr;
#define API_LIST(F) \
 F(HMS_HiAISingleOpTensorDesc_Create) F(HMS_HiAISingleOpTensorDesc_Destroy) \
 F(HMS_HiAISingleOpTensor_CreateFromTensorDesc) F(HMS_HiAISingleOpTensor_CreateFromConst) F(HMS_HiAISingleOpTensor_Destroy) F(HMS_HiAISingleOpTensor_GetBuffer) \
 F(HMS_HiAISingleOpBuffer_Create) F(HMS_HiAISingleOpBuffer_Destroy) F(HMS_HiAISingleOpBuffer_GetData) \
 F(HMS_HiAISingleOpOptions_Create) F(HMS_HiAISingleOpOptions_Destroy) \
 F(HMS_HiAISingleOpDescriptor_CreateConvolution) F(HMS_HiAISingleOpDescriptor_Destroy) \
 F(HMS_HiAISingleOpExecutor_CreateConvolution) F(HMS_HiAISingleOpExecutor_PreCheckConvolution) F(HMS_HiAISingleOpExecutor_GetWorkspaceSize) \
 F(HMS_HiAISingleOpExecutor_Init) F(HMS_HiAISingleOpExecutor_Execute) F(HMS_HiAISingleOpExecutor_Destroy)
#define DECLARE(name) decltype(&name) name=nullptr;
 API_LIST(DECLARE)
#undef DECLARE
 Api(){library=dlopen("libhiai_foundation.so",RTLD_NOW|RTLD_LOCAL);check(library,"CANN public runtime unavailable");
#define LOAD(name) name=reinterpret_cast<decltype(name)>(dlsym(library,#name));check(name,"Missing " #name);
 API_LIST(LOAD)
#undef LOAD
 }
 ~Api(){if(library)dlclose(library);}
};
struct Resources {
 Api& a;HiAI_SingleOpOptions* options=nullptr;HiAI_SingleOpDescriptor* op=nullptr;HiAI_SingleOpExecutor* executor=nullptr;HiAI_SingleOpBuffer* workspace=nullptr;
 std::vector<HiAI_SingleOpTensorDesc*> descriptions;std::vector<HiAI_SingleOpTensor*> tensors;
 explicit Resources(Api& api):a(api){}
 ~Resources(){if(executor)a.HMS_HiAISingleOpExecutor_Destroy(&executor);if(workspace)a.HMS_HiAISingleOpBuffer_Destroy(&workspace);for(auto* t:tensors)a.HMS_HiAISingleOpTensor_Destroy(&t);if(op)a.HMS_HiAISingleOpDescriptor_Destroy(&op);if(options)a.HMS_HiAISingleOpOptions_Destroy(&options);for(auto* d:descriptions)a.HMS_HiAISingleOpTensorDesc_Destroy(&d);}
};
static Json cpu(const std::string& dir,const Json& item){
 Json row={{"success",false},{"backend","MNN CPU Low"},{"threads",6},{"power","High"}};
 try{std::string id=item["id"];auto input=read(dir+"/"+id+"-input.f32"),reference=read(dir+"/"+id+"-reference.f32");
  std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/"+id+".mnn").c_str()));check(bool(net),"CPU graph missing");net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
  MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
  MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;auto* session=net->createSession(config);check(session,"CPU session failed");net->releaseModel();
  auto* x=net->getSessionInput(session,"x");auto* y=net->getSessionOutput(session,"y");check(x&&y,"CPU tensors missing");MNN::Tensor hostX(x,MNN::Tensor::CAFFE,true),hostY(y,MNN::Tensor::CAFFE,true);check(hostX.elementSize()==input.size()&&hostY.elementSize()==reference.size(),"CPU tile shape mismatch");
  for(int i=0;i<4;++i){auto begin=Clock::now();std::memcpy(hostX.host<float>(),input.data(),input.size()*4);check(x->copyFromHostTensor(&hostX),"CPU input copy failed");check(net->runSession(session)==MNN::NO_ERROR,"CPU run failed");check(y->copyToHostTensor(&hostY),"CPU output copy failed");row["runIncludingCopiesSeconds"].push_back(elapsed(begin));}
  double error=0,energy=0;for(size_t i=0;i<reference.size();++i){float value=hostY.host<float>()[i];check(std::isfinite(value),"CPU nonfinite");double d=value-reference[i];error+=d*d;energy+=double(reference[i])*reference[i];}row["relativeL2"]=std::sqrt(error/std::max(energy,1e-30));row["success"]=true;net->releaseSession(session);
 }catch(const std::exception& e){row["error"]=e.what();}return row;
}
static Json run(const std::string& dir,const Json& item){
 Json result=item;result["success"]=false;result["wholeGraphNpuPlacementVerified"]=false;result["matchedCpu"]=cpu(dir,item);
 try{Api a;Resources r(a);std::string id=item["id"];
  auto input=read(dir+"/"+id+"-input.f32"),weight=read(dir+"/"+id+"-weight.f32"),bias=read(dir+"/"+id+"-bias.f32"),reference=read(dir+"/"+id+"-reference.f32");
  std::vector<__fp16> in(input.begin(),input.end()),w(weight.begin(),weight.end()),b(bias.begin(),bias.end()),out(reference.size());
  auto desc=[&](std::vector<int64_t> shape){auto* d=a.HMS_HiAISingleOpTensorDesc_Create(shape.data(),shape.size(),HIAI_SINGLEOP_DT_FLOAT16,HIAI_SINGLEOP_FORMAT_NCHW,false);check(d,"Tensor descriptor failed");r.descriptions.push_back(d);return d;};
  auto tensor=[&](HiAI_SingleOpTensor* t){check(t,"Tensor allocation failed");r.tensors.push_back(t);return t;};
  auto* inputDesc=desc(item["inputShape"].get<std::vector<int64_t>>());auto* outputDesc=desc(item["outputShape"].get<std::vector<int64_t>>());
  auto* wd=desc(item["weightShape"].get<std::vector<int64_t>>());auto* bd=desc({128});
  auto* wt=tensor(a.HMS_HiAISingleOpTensor_CreateFromConst(wd,w.data(),w.size()*2));auto* bt=tensor(a.HMS_HiAISingleOpTensor_CreateFromConst(bd,b.data(),b.size()*2));
  r.options=a.HMS_HiAISingleOpOptions_Create();check(r.options,"Options failed");HiAISingleOpDescriptor_ConvolutionParam conv{};
  conv.convMode=item["operation"]=="Conv"?HIAI_SINGLEOP_CONV_MODE_COMMON:HIAI_SINGLEOP_CONV_MODE_TRANSPOSED;
  conv.strides[0]=conv.dilations[0]=1;conv.strides[1]=item["stride"].get<int>();conv.dilations[1]=item["dilation"].get<int>();conv.pads[2]=conv.pads[3]=item["padding"].get<int>();conv.padMode=HIAI_SINGLEOP_PAD_MODE_SPECIFIC;conv.groups=1;
  r.op=a.HMS_HiAISingleOpDescriptor_CreateConvolution(conv);check(r.op,"Convolution descriptor failed");
  HiAI_SingleOpExecutorConvolutionParam params{r.options,r.op,inputDesc,outputDesc,wt,bt};result["preCheckStatus"]=int(a.HMS_HiAISingleOpExecutor_PreCheckConvolution(params));auto begin=Clock::now();
  r.executor=a.HMS_HiAISingleOpExecutor_CreateConvolution(params);check(r.executor,"NPU executor creation failed");size_t bytes=a.HMS_HiAISingleOpExecutor_GetWorkspaceSize(r.executor);void* workspace=nullptr;
  if(bytes){r.workspace=a.HMS_HiAISingleOpBuffer_Create(bytes);check(r.workspace,"Workspace allocation failed");workspace=a.HMS_HiAISingleOpBuffer_GetData(r.workspace);}
  result["workspaceBytes"]=bytes;check(a.HMS_HiAISingleOpExecutor_Init(r.executor,workspace,bytes)==OH_NN_SUCCESS,"NPU init failed");result["setupSeconds"]=elapsed(begin);
  auto* it=tensor(a.HMS_HiAISingleOpTensor_CreateFromTensorDesc(inputDesc));auto* ot=tensor(a.HMS_HiAISingleOpTensor_CreateFromTensorDesc(outputDesc));
  auto* ip=a.HMS_HiAISingleOpBuffer_GetData(a.HMS_HiAISingleOpTensor_GetBuffer(it));auto* op=a.HMS_HiAISingleOpBuffer_GetData(a.HMS_HiAISingleOpTensor_GetBuffer(ot));check(ip&&op,"Tensor mapping failed");
  for(int i=0;i<4;++i){begin=Clock::now();std::memcpy(ip,in.data(),in.size()*2);check(a.HMS_HiAISingleOpExecutor_Execute(r.executor,&it,1,&ot,1)==OH_NN_SUCCESS,"NPU execute failed");std::memcpy(out.data(),op,out.size()*2);result["runIncludingCopiesSeconds"].push_back(elapsed(begin));}
  double error=0,energy=0;for(size_t i=0;i<out.size();++i){check(std::isfinite(float(out[i])),"Nonfinite output");double d=float(out[i])-reference[i];error+=d*d;energy+=double(reference[i])*reference[i];}
  result["relativeL2"]=std::sqrt(error/std::max(energy,1e-30));result["success"]=result["relativeL2"].get<double>()<.02;
 }catch(const std::exception& e){result["error"]=e.what();}return result;
}
int main(int argc,char** argv){if(argc!=2)return 2;try{Json manifest;std::ifstream(argv[1]+std::string("/manifest.json"))>>manifest;Json report={{"scope",manifest["scope"]},{"fullVaeOnNpu",false},{"cases",Json::array()}};for(auto& item:manifest["cases"]){auto row=run(argv[1],item);report["cases"].push_back(row);std::cout<<"LEVO_NPU_PROGRESS "<<row.dump()<<std::endl;}std::cout<<"LEVO_NPU_REPORT "<<report.dump()<<std::endl;return 0;}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
