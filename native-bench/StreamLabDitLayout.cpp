// Current-bundle layout audit and first real FFN capture, not a replacement model.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/TensorUtils.hpp"
#include <MNN_generated.h>
using namespace LevoStreamLab;
namespace {
void save(const std::string& path,const std::vector<float>& values){std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(values.data()),values.size()*4);require(out.good(),"Cannot write capture");}
Json tensors(const std::vector<MNN::Tensor*>& ts){
    Json result=Json::array();for(auto t:ts){auto desc=MNN::TensorUtils::getDescribe(t);result.push_back({{"shape",t->shape()},{"logicalTypeBits",t->getType().bits},{"format",MNN::EnumNameMNN_DATA_FORMAT(desc->dimensionFormat)}});}return result;
}
Json capture(const std::string& dir,const std::string& file,MNN::Tensor* tensor){
    MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(host.getType().bits==32&&tensor->copyToHostTensor(&host),"FP32 intermediate capture failed");
    std::vector<float> values(host.host<float>(),host.host<float>()+host.elementSize());peak(values);save(dir+"/"+file,values);return {{"file",file},{"hostShape",host.shape()},{"values",values.size()}};
}
Json decode(const MNN::Op* source,const std::string& weightPath,const std::string& output,const std::string& label){
    std::unique_ptr<MNN::OpT> op(source->UnPack());op->externalPath=weightPath;auto conv=op->main.AsConvolution2D();require(conv&&conv->common->kernelX==1&&conv->common->kernelY==1&&conv->common->group==1,"Only verified 1x1 projections supported");
    flatbuffers::FlatBufferBuilder builder;builder.Finish(MNN::Op::Pack(builder,op.get()));auto decoded=flatbuffers::GetRoot<MNN::Op>(builder.GetBufferPointer());
    auto weights=MNN::ConvolutionCommon::load(decoded,nullptr,true,false);require(weights&&weights->weightFloat.get(),"Cannot decode original W8 parameters");
    std::vector<float> w(weights->weightFloat.get(),weights->weightFloat.get()+weights->weightFloat.size()),b=conv->bias;require(w.size()==size_t(conv->common->inputCount)*conv->common->outputCount,"Decoded projection shape changed");
    if(b.empty()){const auto& e=conv->external;require(e.size()>=4&&e[3]==conv->common->outputCount*4,"Missing real bias metadata");b.resize(conv->common->outputCount);std::ifstream in(weightPath,std::ios::binary);in.seekg(e[0]+e[1]+e[2]);in.read(reinterpret_cast<char*>(b.data()),e[3]);require(in.good(),"Missing real bias bytes");}
    save(output+"/"+label+"-weight.f32",w);save(output+"/"+label+"-bias.f32",b);
    return {{"name",op->name},{"inputChannels",conv->common->inputCount},{"outputChannels",conv->common->outputCount},{"weightShape",{conv->common->outputCount,conv->common->inputCount}},{"weightFile",label+"-weight.f32"},{"biasFile",label+"-bias.f32"},{"source","Decoded original MNN quantized weights; not original FP32 checkpoint"}};
}
}
int main(int argc,char** argv){
    if(argc<4||argc>6)return 2;bool cfg=true,normalMemory=false;
    for(int i=4;i<argc;++i){std::string flag=argv[i];if(flag=="--no-cfg")cfg=false;else if(flag=="--memory-normal")normalMemory=true;else return 2;}
    const std::string root=argv[1],output=argv[3],model=root+"/dit/dit.mnn";std::filesystem::create_directories(output);
    Json report={{"success",false},{"scope","Current22-layer W8 DiT layout and captured first FFN; callback/capture overhead, not performance benchmark"},{"batch",2},{"frames",256},{"hidden",1200},{"ffn",4800},{"cfgBranches",cfg?"unconditional+conditional":"conditional+conditional"},{"backend",normalMemory?"MNN CPU High / Memory Normal / 6threads":"MNN CPU High / Memory Low / 6threads"}};
    try{
        if(normalMemory)require(available()>3000*1024,"Less than3000MiB system available for high-precision diagnostic");
        std::ifstream in(model,std::ios::binary|std::ios::ate);require(bool(in),"Missing current graph");std::vector<char> bytes(size_t(in.tellg()));in.seekg(0);in.read(bytes.data(),bytes.size());
        auto netGraph=MNN::GetNet(bytes.data());const MNN::Op *up=nullptr,*down=nullptr;std::map<std::string,int> counts;Json staticOps=Json::array();
        for(auto source:*netGraph->oplists()){
            std::unique_ptr<MNN::OpT> op(source->UnPack());std::string type=MNN::EnumNameOpType(op->type);++counts[type];Json row={{"name",op->name},{"type",type},{"inputs",op->inputIndexes},{"outputs",op->outputIndexes}};
            if(auto c=op->main.AsConvolution2D()){row["conv"]={{"inputChannels",c->common->inputCount},{"outputChannels",c->common->outputCount},{"kernel",{c->common->kernelY,c->common->kernelX}},{"quantized",bool(c->quanParameter)}};
                if(!up&&c->common->inputCount==1200&&c->common->outputCount==4800)up=source;
                if(up&&!down&&c->common->inputCount==4800&&c->common->outputCount==1200)down=source;}
            if(auto p=op->main.AsPermute())row["perm"]=p->dims;
            if(auto r=op->main.AsReshape())row["reshape"]=r->dims;
            if(auto b=op->main.AsBlob())if(!b->int32s.empty()&&b->int32s.size()<=12)row["smallInts"]=b->int32s;
            staticOps.push_back(row);
        }
        require(up&&down,"Could not identify the current first1200/4800 FFN pair");report["staticOperatorCounts"]=counts;report["staticGraph"]=staticOps;
        const std::string upName=up->name()->str(),downName=down->name()->str();report["up"]=decode(up,model+".weight",output,"up");report["down"]=decode(down,model+".weight",output,"down");
        auto codes=readJson(argv[2]).get<std::vector<int>>();require(codes.size()>=256,"Need first256 actual music codes");MappedFloats table(root+"/conditioner/cond-token.f16");
        std::vector<float> mu(256*1024);for(int i=0;i<256;++i){require(codes[i]>=0&&codes[i]<16384,"Invalid music code");table.row(codes[i],1024,mu.data()+i*1024);}
        auto masks=read(root+"/aux/mask.f32",144);std::mt19937 rng(2307);std::normal_distribution<float> normal;std::vector<float> x(256*64);for(auto& v:x)v=normal(rng);
        std::vector<float> inputs(2*256*1200,0),attention(2*256*256,1.f);for(int b=0;b<2;++b)for(int i=0;i<256;++i){size_t at=size_t(b*256+i)*1200;std::copy_n(masks.data()+96,48,inputs.data()+at);if(!cfg||b)std::copy_n(mu.data()+i*1024,1024,inputs.data()+at+112);std::copy_n(x.data()+i*64,64,inputs.data()+at+1136);}
        std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model.c_str()));net->setExternalFile((model+".weight").c_str());net->setSessionMode(MNN::Interpreter::Session_Debug);net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
        MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_High;backend.memory=normalMemory?MNN::BackendConfig::Memory_Normal:MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;auto session=net->createSession(config);require(session,"Cannot create audit session");net->resizeSession(session);net->releaseModel();
        auto put=[&]{for(auto& pair:std::map<std::string,std::vector<float>>{{"inputs",inputs},{"timestep",{0,0}},{"attention_mask",attention}}){auto tensor=net->getSessionInput(session,pair.first.c_str());require(tensor,"Missing audit input");MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(host.elementSize()==pair.second.size(),"Audit input count mismatch");std::memcpy(host.host<float>(),pair.second.data(),pair.second.size()*4);require(tensor->copyFromHostTensor(&host),"Audit input copy failed");}};
        put();require(net->runSession(session)==0,"Audit warm run failed");put();Json nodes=Json::array();Clock::time_point tick;Json current;
        auto before=[&](const std::vector<MNN::Tensor*>& ts,const MNN::OperatorInfo* info){current={{"name",info->name()},{"type",info->type()},{"inputs",tensors(ts)}};
            if(info->name()==upName)report["upInput"]=capture(output,"up-input.f32",ts[0]);if(info->name()==downName)report["downInput"]=capture(output,"down-input.f32",ts[0]);tick=Clock::now();return true;};
        auto after=[&](const std::vector<MNN::Tensor*>& ts,const MNN::OperatorInfo* info){current["seconds"]=seconds(tick);current["outputs"]=tensors(ts);
            if(info->name()==upName)report["upOutput"]=capture(output,"up-output.f32",ts[0]);if(info->name()==downName)report["downOutput"]=capture(output,"down-output.f32",ts[0]);nodes.push_back(current);return true;};
        require(net->runSessionWithCallBackInfo(session,before,after,true)==0,"Audit callback run failed");report["runtimeNodes"]=nodes;
        auto tensor=net->getSessionOutput(session,nullptr);MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(tensor->copyToHostTensor(&host),"Audit velocity copy failed");report["velocityPeak"]=peak(std::vector<float>(host.host<float>(),host.host<float>()+host.elementSize()));
        require(report.contains("upInput")&&report.contains("downOutput"),"First FFN capture callbacks not found");net->releaseSession(session);report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    std::ofstream(output+"/layout.json")<<report.dump(2);std::cout<<"DIT_LAYOUT_DONE "<<report.value("success",false)<<std::endl;return report["success"]==true?0:1;
}
