// Isolated current-bundle VAE window experiment. Does not modify either App.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include <numeric>
using namespace LevoStreamLab;

namespace {
constexpr int channels=64, frames=256, rate=1920;
void saveFloats(const std::string& path,const std::vector<float>& data){
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),data.size()*4);
    require(out.good(),"Cannot save numerical fixture");
}
std::vector<float> crop(const std::vector<float>& values,int start,int count){
    require(start>=0&&count>0&&start+count<=frames,"Invalid latent window");
    std::vector<float> out(channels*count);
    for(int c=0;c<channels;++c)std::copy_n(values.data()+c*frames+start,count,out.data()+c*count);
    return out;
}
Json errorRange(const std::vector<float>& actual,const std::vector<float>& reference,int offset,int length,int begin,int end){
    require(begin>=offset&&end<=offset+length&&begin<end,"Invalid PCM comparison interval");
    require(reference.size()==2*frames*rate&&actual.size()==size_t(2)*length*rate,"PCM reference shape mismatch");
    double energy=0,error=0,maximum=0;size_t n=0;
    for(int c=0;c<2;++c)for(int i=begin*rate;i<end*rate;++i){
        double a=actual[size_t(c)*length*rate+i-offset*rate],b=reference[size_t(c)*frames*rate+i];
        require(std::isfinite(a)&&std::isfinite(b),"Non-finite PCM");
        error+=(a-b)*(a-b);energy+=b*b;maximum=std::max(maximum,std::abs(a-b));++n;
    }
    return {{"relativeL2",std::sqrt(error/std::max(energy,1e-30))},{"maxAbsoluteDiff",maximum},{"samples",n},
            {"beginFrame",begin},{"endFrame",end},{"passes1eMinus3",std::sqrt(error/std::max(energy,1e-30))<=1e-3}};
}
class Decoder {
    std::unique_ptr<MNN::Interpreter> net_;MNN::Session* session_=nullptr;MNN::Tensor* input_=nullptr;
public:
    Json timing;int length;
    Decoder(const std::string& path,int count,bool debug=false,bool retainGraph=false):length(count){
        auto tick=Clock::now();net_.reset(MNN::Interpreter::createFromFile(path.c_str()));require(bool(net_),"Cannot load current VAE");
        auto external=path+".weight";net_->setExternalFile(external.c_str());
        net_->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        if(debug)net_->setSessionMode(MNN::Interpreter::Session_Debug);
        net_->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
        MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;
        session_=net_->createSession(config);require(session_,"Cannot create VAE session");
        input_=net_->getSessionInput(session_,nullptr);require(input_,"Missing VAE input");auto dims=input_->shape();timing["originalInputShape"]=dims;
        require(dims==std::vector<int>({1,64,256}),"Current VAE input contract changed");dims.back()=count;net_->resizeTensor(input_,dims);
        {LevoVaeSin::Scope scope(true);net_->resizeSession(session_);}if(!retainGraph)net_->releaseModel();
        auto output=net_->getSessionOutput(session_,nullptr);require(output&&output->elementSize()==2*count*rate,"Resized VAE output mismatch");
        timing["setupSeconds"]=seconds(tick);timing["outputShape"]=output->shape();timing["rssAfterSetupMiB"]=::resident()/1024.;
    }
    ~Decoder(){if(session_)net_->releaseSession(session_);}
    Json resize(int count){
        auto tick=Clock::now();auto dims=input_->shape();require(dims.size()==3,"Resize input rank changed");dims.back()=count;
        net_->resizeTensor(input_,dims);{LevoVaeSin::Scope scope(true);net_->resizeSession(session_);}
        auto output=net_->getSessionOutput(session_,nullptr);require(output&&output->elementSize()==2*count*rate,"Reused session output mismatch");length=count;
        return {{"seconds",seconds(tick)},{"outputShape",output->shape()},{"rssMiB",::resident()/1024.}};
    }
    std::vector<float> run(const std::vector<float>& latent,Json& row,bool profile=false){
        check();auto total=Clock::now(),tick=total;
        MNN::Tensor hostInput(input_,MNN::Tensor::CAFFE,true);require(hostInput.elementSize()==latent.size(),"Input shape mismatch");
        std::memcpy(hostInput.host<float>(),latent.data(),latent.size()*4);require(input_->copyFromHostTensor(&hostInput),"Input transfer failed");
        row["inputSeconds"]=seconds(tick);tick=Clock::now();
        if(profile){
            std::map<std::string,Json> nodes;Clock::time_point opStart;
            auto before=[&](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo*){opStart=Clock::now();return true;};
            auto after=[&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* info){
                double elapsed=seconds(opStart);auto& node=nodes[info->name()];if(node.is_null())node={{"name",info->name()},{"type",info->type()},{"seconds",0.},{"calls",0}};
                node["seconds"]=node["seconds"].get<double>()+elapsed;node["calls"]=node["calls"].get<int>()+1;
                node["outputShapes"]=Json::array();for(auto tensor:tensors)node["outputShapes"].push_back(tensor->shape());return true;
            };
            require(net_->runSessionWithCallBackInfo(session_,before,after,true)==MNN::NO_ERROR,"Profile failed");
            std::map<std::string,double> types;Json ordered=Json::array();for(auto& pair:nodes){ordered.push_back(pair.second);types[pair.second["type"].get<std::string>()]+=pair.second["seconds"].get<double>();}
            std::sort(ordered.begin(),ordered.end(),[](const Json& a,const Json& b){return a["seconds"].get<double>()>b["seconds"].get<double>();});
            row["byTypeSeconds"]=types;row["operators"]=ordered;
        }else require(net_->runSession(session_)==MNN::NO_ERROR,"VAE forward failed");
        row["forwardSeconds"]=seconds(tick);tick=Clock::now();
        auto output=net_->getSessionOutput(session_,nullptr);MNN::Tensor host(output,MNN::Tensor::CAFFE,true);
        require(output->copyToHostTensor(&host),"Output transfer failed");std::vector<float> pcm(host.host<float>(),host.host<float>()+host.elementSize());
        row["outputSeconds"]=seconds(tick);row["totalSeconds"]=seconds(total);row["peak"]=peak(pcm);row["rssMiB"]=::resident()/1024.;return pcm;
    }
};
void prepare(const std::string& root,const std::string& codesPath,const std::string& output,bool lastWindow){
    auto codes=readJson(codesPath).get<std::vector<int>>();require(codes.size()>=256&&codes.size()<=1500,"Need256–1500 real sampled music codes");
    MappedFloats table(root+"/conditioner/cond-token.f16");std::vector<float> mu(frames*1024);
    auto masks=read(root+"/aux/mask.f32",144),mean=read(root+"/aux/mean.f32",64),deviation=read(root+"/aux/std.f32",64);
    std::mt19937 rng(2307);std::normal_distribution<float> normal;std::vector<float> x(frames*64),previous;
    // Match the App's early final-window break, including exact window ends.
    const int lastStart=lastWindow?((std::max(0,int(codes.size())-frames)+191)/192)*192:0;int lastValid=256;
    {
        Graph dit(root+"/dit/dit.mnn",false,6,false,2);
        for(int start=0;start<=lastStart;start+=192){
        const int produced=std::min(int(codes.size()),start+256),valid=std::min(256,int(codes.size())-start);lastValid=valid;
        for(int i=0;i<frames;++i){int code=codes[(start+i)%produced];require(code>=0&&code<16384,"Music code outside codebook");table.row(code,1024,mu.data()+i*1024);}
        std::vector<float> context(frames*64,0),noise(frames*64);if(start)std::copy_n(previous.data()+192*64,64*64,context.data());for(float& v:noise)v=normal(rng);x=noise;
        for(int step=0;step<2;++step){
            const float t=float(step)/2;if(start)for(int j=0;j<64*64;++j)x[j]=(1-(1-1e-4f)*t)*noise[j]+t*context[j];
            std::vector<float> input(2*frames*1200,0),attention(2*frames*frames,0.f);
            for(int b=0;b<2;++b)for(int i=0;i<frames;++i){size_t offset=size_t(b*frames+i)*1200;
                const int mask=i>=valid?0:(start&&i<64?1:2);std::copy_n(masks.data()+mask*48,48,input.data()+offset);std::copy_n(context.data()+i*64,64,input.data()+offset+48);
                std::copy_n(mu.data()+i*1024,1024,input.data()+offset+112);std::copy_n(x.data()+i*64,64,input.data()+offset+1136);
                if(i<valid)std::fill_n(attention.data()+size_t(b*frames+i)*frames,valid,1.f);}
            auto v=dit.run({{"inputs",std::move(input)},{"attention_mask",std::move(attention)},{"timestep",std::vector<float>(2,t)}},2*frames*64);
            for(size_t i=0;i<x.size();++i)x[i]+=v[i]/2.f;peak(x);
        }
        if(start)std::copy_n(context.data(),64*64,x.data());previous=x;
        std::cout<<"VAE_FIXTURE_DIT "<<start/192<<std::endl;
        }
    }
    std::vector<float> latent(64*frames);for(int c=0;c<64;++c)for(int i=0;i<frames;++i)latent[c*frames+i]=x[i*64+c]*deviation[c]+mean[c];
    saveFloats(output+"/latent.f32",latent);std::ofstream(output+"/fixture.json")<<Json({{"sourceCodes",codesPath},{"sourceBundle",root},
        {"scope","Real sampled codes reconstructed through current22-layer DiT,2steps,no CFG mixing,batch2; LLM not run in this diagnostic"},
        {"startFrame",lastStart},{"validFrames",lastValid},{"seed",2307},{"latentPeak",peak(latent)},{"frames",256}}).dump(2);
}
}
int main(int argc,char** argv){
    if(argc!=5){std::cerr<<"bundleDir codes.json outputDir prepare|prepare-tail|windows|confirm|profile|reuse|reuse-kept\n";return 2;}
    const std::string root=argv[1],out=argv[3],mode=argv[4];std::filesystem::create_directories(out);Json report={{"success",false},{"mode",mode},{"backend","CPU Low"},{"threadsRequested",6},
        {"vectorSin",true},{"weightRecipe","Original immutable bundle W8/g64"},{"sampleIntervalMs",20},{"wholeChainTest",false},{"speakerPlaybackTested",false}};
    Trace trace;
    try{
        if(mode=="prepare"||mode=="prepare-tail"){prepare(root,argv[2],out,mode=="prepare-tail");report["fixture"]=readJson(out+"/fixture.json");}
        else{
            require(mode=="windows"||mode=="confirm"||mode=="profile"||mode=="reuse"||mode=="reuse-kept","Unknown diagnostic mode");
            auto latent=read(out+"/latent.f32",64*frames);report["fixture"]=readJson(out+"/fixture.json");
            if(mode=="reuse"||mode=="reuse-kept"){
                const bool retained=mode=="reuse-kept";report["graphBufferRetained"]=retained;
                Decoder decoder(root+"/vae/vae.mnn",256,false,retained);Json first;auto reference=decoder.run(latent,first);report["initial"]=first;
                report["resizeTo172"]=decoder.resize(172);auto input=crop(latent,0,172);report["shortRuns"]=Json::array();
                for(int i=0;i<3;++i){Json row;auto pcm=decoder.run(input,row);row["comparison"]=errorRange(pcm,reference,0,172,0,156);report["shortRuns"].push_back(row);}
                report["restore256"]=decoder.resize(256);Json restored;auto pcm=decoder.run(latent,restored);restored["comparison"]=errorRange(pcm,reference,0,256,0,256);report["restored"]=restored;
                report["oneVaeSessionOnly"]=true;
            }else if(mode=="profile"){
                Decoder decoder(root+"/vae/vae.mnn",256,true);Json warm,profile;decoder.run(latent,warm);decoder.run(latent,profile,true);
                report["warm"]=warm;report["profile"]=profile;report["setup"]=decoder.timing;
            }else{
                struct Case{const char* name;int offset,length,begin,end;};
                std::vector<Case> cases={{"baseline",0,256,0,256},{"tail-no-halo",0,156,0,156},{"tail-right8",0,164,0,156},
                    {"tail-right16",0,172,0,156},{"new-region-left16",48,208,64,256},{"baseline-repeat",0,256,0,256}};
                if(mode=="confirm")cases={{"baseline",0,256,0,256},{"tail-right16",0,172,0,156},{"new-region-left16",48,208,64,256},{"baseline-repeat",0,256,0,256}};
                Json rows=Json::array();std::vector<float> reference;
                for(const auto& c:cases){
                    check();Json row={{"case",c.name},{"offset",c.offset},{"frames",c.length},{"runs",Json::array()}};
                    try{
                        require((c.offset==0&&c.length==256)||!reference.empty(),"Baseline failed; no independent full-window reference");
                        Decoder decoder(root+"/vae/vae.mnn",c.length);row["setup"]=decoder.timing;Json warm;auto input=crop(latent,c.offset,c.length);decoder.run(input,warm);row["warm"]=warm;
                        double sum=0;for(int repeat=0;repeat<2;++repeat){Json sample;auto pcm=decoder.run(input,sample);sum+=sample["totalSeconds"].get<double>();
                            if(reference.empty()){reference=pcm;saveFloats(out+"/reference.f32",pcm);}
                            sample["comparison"]=errorRange(pcm,reference,c.offset,c.length,c.begin,c.end);row["runs"].push_back(sample);}
                        row["meanSeconds"]=sum/2;row["success"]=true;
                    }catch(const std::exception& e){row["success"]=false;row["error"]=e.what();}
                    trace.sample();rows.push_back(row);report["cases"]=rows;std::ofstream(out+"/"+mode+".json")<<report.dump(2);
                    std::cout<<"VAE_WINDOW_CASE "<<row.dump()<<std::endl;
                }
            }
        }
        report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["minimumSystemAvailableMiB"]=trace.minAvailable.load()/1024.;report["elapsedSeconds"]=seconds(trace.start);
    std::ofstream(out+"/"+mode+".json")<<report.dump(2);std::cout<<"VAE_WINDOW_REPORT "<<report.dump()<<std::endl;return report["success"]==true?0:1;
}
