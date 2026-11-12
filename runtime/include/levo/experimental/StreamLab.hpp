#pragma once
// Independent numerical streaming experiment. Never included by either old App.
#include "../RuntimeSupport.hpp"
#include "HiddenStateTap.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <regex>
#include "tokenizer/tokenizer.hpp"

namespace LevoStreamLab {
inline std::atomic<bool> cancelled{false},busy{false};
inline std::mutex stateMutex;
inline Json state={{"busy",false},{"status","idle"},{"chunks",Json::array()},{"generatedSamples",0},{"requestedFrames",0}};
inline Json snapshot(){std::lock_guard<std::mutex> lock(stateMutex);return state;}
inline void patch(const Json& value){std::lock_guard<std::mutex> lock(stateMutex);for(auto it=value.begin();it!=value.end();++it)state[it.key()]=it.value();}
inline void check(){require(!cancelled.load(),"Generation cancelled");require(::resident()<6L*1024*1024,"Process RSS reached6GiB; published prefix is retained");require(available()>650*1024,"System available memory below650MiB; published prefix is retained");}
inline std::string modelId(int choice){require(choice==0||choice==1,"Unknown model");return choice==0?"larkstudio-small-0.6b-v1":"larkstudio-base-v2";}
inline Json inspect(const std::string& root,int choice){
    Json r={{"ready",false},{"modelId",modelId(choice)},{"count",0},{"total",0}};
    try{auto m=readJson(root+"/model-bundles/"+modelId(choice)+"/model-bundle.json");
        auto l=m["modules"]["llm"]["recipe"],d=m["modules"]["dit"]["recipe"];
        require(l["layers"]==(choice==0?28:36)&&l["hidden"]==(choice==0?1024:2560)&&l["intermediate"]==(choice==0?3072:9728),"D1 architecture differs from requested specification");
        require(d["layers"]==(choice==0?22:39)&&d["hidden"]==1200&&d["intermediate"]==4800&&d["heads"]==30,"D2 architecture differs from requested specification");
        int count=0,total=0;for(const auto& a:m["assets"]){++total;auto p=std::filesystem::path(root)/"model-bundles"/modelId(choice)/a["relative"].get<std::string>();
            if(std::filesystem::is_regular_file(p)&&std::filesystem::file_size(p)==a["bytes"].get<uint64_t>())++count;}
        r["count"]=count;r["total"]=total;r["ready"]=count==total;r["integrityScope"]="Architecture declaration and file sizes; deployment verifies SHA256";
    }catch(const std::exception& e){r["error"]=e.what();}return r;
}
class MappedFloats {
    int fd_=-1;size_t bytes_=0;void* ptr_=MAP_FAILED;
public:
    explicit MappedFloats(const std::string& path){fd_=open(path.c_str(),O_RDONLY);require(fd_>=0,"Cannot map "+path);struct stat s{};
        require(fstat(fd_,&s)==0,"Cannot stat mapped table");bytes_=s.st_size;ptr_=mmap(nullptr,bytes_,PROT_READ,MAP_PRIVATE,fd_,0);require(ptr_!=MAP_FAILED,"Cannot map table");}
    ~MappedFloats(){if(ptr_!=MAP_FAILED)munmap(ptr_,bytes_);if(fd_>=0)close(fd_);}
    void row(size_t index,int width,float* target,bool bf16=false){require((index+1)*size_t(width)*2<=bytes_,"Embedding row outside table");
        const auto* p=static_cast<const uint16_t*>(ptr_)+index*width;
        for(int c=0;c<width;++c){if(bf16){uint32_t b=uint32_t(p[c])<<16;std::memcpy(target+c,&b,4);}else {__fp16 h;std::memcpy(&h,p+c,2);target[c]=float(h);}}
    }
};
class Graph {
    std::unique_ptr<MNN::Interpreter> net_;MNN::Session* session_=nullptr;
public:
    Graph(const std::string& path,bool low,int threads,bool vae=false,int ditBatch=0){
        net_.reset(MNN::Interpreter::createFromFile(path.c_str()));require(bool(net_),"Cannot load "+path);auto external=path+".weight";net_->setExternalFile(external.c_str());
        net_->setSessionMode(MNN::Interpreter::Session_Resize_Defer);net_->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
        MNN::BackendConfig b;b.precision=low?MNN::BackendConfig::Precision_Low:MNN::BackendConfig::Precision_High;b.memory=MNN::BackendConfig::Memory_Low;b.power=MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig cfg;cfg.type=MNN_FORWARD_CPU;cfg.numThread=threads;cfg.backendConfig=&b;session_=net_->createSession(cfg);require(session_,"Cannot create model session");
        if(ditBatch){require(ditBatch==1||ditBatch==2,"Invalid DiT batch");
            auto inputs=net_->getSessionInput(session_,"inputs"),timestep=net_->getSessionInput(session_,"timestep"),mask=net_->getSessionInput(session_,"attention_mask");
            require(inputs&&timestep&&mask,"DiT batch resize input mismatch");
            for(auto tensor:{inputs,timestep,mask}){auto dims=tensor->shape();std::cout<<"STREAM_GRAPH_INPUT "<<Json(dims).dump()<<std::endl;
                if(ditBatch==1){require(!dims.empty()&&dims[0]==2,"Cannot identify batch axis");dims[0]=1;net_->resizeTensor(tensor,dims);}}}
        LevoVaeSin::Scope scope(vae);net_->resizeSession(session_);require(net_->getSessionOutput(session_,nullptr),"Graph resize failed");net_->releaseModel();
    }
    ~Graph(){if(session_)net_->releaseSession(session_);}
    std::vector<float> run(const std::map<std::string,std::vector<float>>& inputs,size_t expected){
        for(const auto& p:inputs){auto tensor=p.first.empty()?net_->getSessionInput(session_,nullptr):net_->getSessionInput(session_,p.first.c_str());require(tensor,"Input name mismatch: "+p.first);
            MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(size_t(host.elementSize())==p.second.size(),"Input size mismatch");std::memcpy(host.host<float>(),p.second.data(),p.second.size()*4);require(tensor->copyFromHostTensor(&host),"Input transfer failed");}
        check();require(net_->runSession(session_)==MNN::NO_ERROR,"Model forward failed");auto tensor=net_->getSessionOutput(session_,nullptr);require(tensor,"Missing graph output");
        MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(size_t(host.elementSize())==expected&&tensor->copyToHostTensor(&host),"Output contract mismatch");
        std::vector<float> value(host.host<float>(),host.host<float>()+expected);peak(value);return value;
    }
};
inline std::vector<float> npy512(const std::string& path){
    std::ifstream in(path,std::ios::binary|std::ios::ate);require(in&&in.tellg()<65536,"NPY文件不可读或过大");in.seekg(0);char magic[8];in.read(magic,8);
    require(std::string(magic,6)==std::string("\x93NUMPY",6),"请选择NumPy .npy文件");uint32_t length=0;
    const int bytes=magic[6]==1?2:4;require(magic[6]>=1&&magic[6]<=3,"不支持此NPY版本");
    for(int i=0;i<bytes;++i){unsigned char x=0;in.read(reinterpret_cast<char*>(&x),1);length|=uint32_t(x)<<(8*i);}require(length<8192,"NPY头过长");
    std::string head(length,0);in.read(head.data(),length);
    require(head.find("<f4")!=std::string::npos&&head.find("True")==std::string::npos,"NPY需为little-endian float32");
    require(std::regex_search(head,std::regex("['\"]shape['\"]\\s*:\\s*\\(\\s*(1\\s*,\\s*)?512\\s*,?\\s*\\)")),"NPY形状需为[512]或[1,512]");
    std::vector<float> values(512);in.read(reinterpret_cast<char*>(values.data()),2048);require(in.gcount()==2048&&in.peek()==std::char_traits<char>::eof(),"NPY长度不匹配512维");peak(values);return values;
}
inline std::string normalizeLyrics(std::string text){size_t p=0;while((p=text.find("\\n",p))!=std::string::npos){text.replace(p,2,"\n");++p;}return text;}
inline std::vector<float> prefix(const std::string& root,int width,bool small,const std::string& lyrics="",const std::string& npy=""){
    auto start=read(root+"/conditioner/start.f32",width),weight=read(root+"/conditioner/genre2start.f32",size_t(width)*768);
    auto artist=read(root+"/conditioner/artist.f32",1024);std::vector<float> style(768,0);
    auto clap=npy512(npy.empty()?root+"/clap/default.npy":npy);std::copy(clap.begin(),clap.end(),style.begin());
    std::copy_n(artist.data(),256,style.data()+512);
    MappedFloats text(root+(small?"/lm/embeddings_bf16.bin":"/conditioner/x_emb.f16"));
    std::vector<float> values(size_t(513)*width);std::copy(start.begin(),start.end(),values.begin());
    std::vector<int> ids;
    if(!lyrics.empty()){
        std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer(MNN::Transformer::Tokenizer::createTokenizer(root+"/conditioner/tokenizer.mtok"));require(bool(tokenizer),"Cannot load lyric tokenizer");
        ids=tokenizer->encode("\n"+normalizeLyrics(lyrics));require(ids.size()<=511,"歌词超过511 token，请缩短，不会自动截断");
        for(int id:ids)require(id>=0&&id<151830,"Lyric ID outside text embedding vocabulary");
        patch({{"lyricTokens",ids.size()},{"lyrics",normalizeLyrics(lyrics)},{"npy",npy},{"samplingMode","temperature0.95-topK250"}});
    }
    for(int t=0;t<511;++t)text.row(lyrics.empty()?100+(t*17)%10000:(t<int(ids.size())?ids[t]:0),width,values.data()+size_t(t+1)*width,small);
    for(int c=0;c<width;++c){double sum=0;for(int k=0;k<768;++k)sum+=weight[size_t(c)*768+k]*style[k];values[size_t(512)*width+c]=float(sum);}
    // Match the original App's generated BF16 input table, including style.
    for(float& value:values){uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);bits&=0xffff0000;std::memcpy(&value,&bits,4);}
    peak(values);return values;
}
class Prior {
    std::unique_ptr<Llm,decltype(&Llm::destroy)> model_{nullptr,&Llm::destroy};
    MNN::Express::VARP logits_;std::unique_ptr<MappedFloats> table_;bool small_,sampling_,smallHidden_,cropped_;int width_,previous_=0;std::mt19937 random_{2307};std::vector<int> codes_;
public:
    Prior(const std::string& root,const std::string& work,bool small,const std::string& lyrics="",const std::string& npy="",bool smallHidden=false,const std::string& croppedLm="",int threads=4):small_(small),sampling_(!lyrics.empty()),smallHidden_(smallHidden),cropped_(!croppedLm.empty()),width_(small?1024:2560){
        require(!cropped_||small,"Music-head crop is for the small model only");
        if(cropped_){auto range=readJson(croppedLm+"/head-range.json");require(range["success"]==true&&range["firstRetainedClass"]==151765&&range["classes"]==16449,"Invalid head-range contract");
            require(std::filesystem::file_size(croppedLm+"/embeddings_bf16.bin")==344502272,"Cropped LM requires the complete input embedding file");}
        Json cfg={{"base_dir",cropped_?croppedLm+"/":root+"/lm/"},{"llm_model","llm.mnn"},{"llm_weight","llm.mnn.weight"},{"llm_config","llm_config.json"},
          {"tokenizer_file","tokenizer.mtok"},{"embedding_file","embeddings_bf16.bin"},{"backend_type","cpu"},{"precision","high"},{"memory","low"},{"power","high"},
          {"thread_num",threads},{"reuse_kv",false},{"attention_mode",8},{"hidden_states",!small||smallHidden},{"all_logits",false},{"cosyvoice_kleidiai",false},{"tmp_path",work+"/"}};
        {std::ofstream out(work+"/lm-config.json");out<<cfg.dump(2);}
        model_.reset(Llm::createLLM(work+"/lm-config.json"));require(model_&&model_->load(),"D1 load failed");check();model_->reset();model_->generate_init();
        patch({{"phase","prefill"}});auto p=prefix(root,width_,small_,lyrics,npy);
        {std::ofstream out(work+"/prefix.f32",std::ios::binary);out.write(reinterpret_cast<const char*>(p.data()),p.size()*4);}
        logits_=model_->forward(MNN::Express::_Const(p.data(),{513,1,width_},MNN::Express::NCHW,halide_type_of<float>()));
        require(logits_.get()&&logits_->readMap<float>(),"Prefill failed");
        if(small_&&!smallHidden_)table_=std::make_unique<MappedFloats>(root+"/conditioner/cond-token.f16");
    }
    std::vector<float> next(int index){
        if(small_&&!smallHidden_&&index)logits_=model_->forward({151830+previous_},false);
        int code=(index*997+2307)%16384;
        if(sampling_){
            const int classes=small_?(cropped_?16449:168214):16385,audioBase=small_?(cropped_?65:151830):1,eos=small_&&!cropped_?151765:0;require(logits_.get()&&logits_->getInfo()->size==classes,"D1输出词表不匹配");const float* p=logits_->readMap<float>();require(p,"Cannot read sampling logits");
            std::vector<std::pair<float,int>> candidates;candidates.reserve(16385);
            for(int k=0;k<16384;++k){float value=p[audioBase+k];require(std::isfinite(value),"Nonfinite sampling logits");candidates.emplace_back(value,k);}
            require(std::isfinite(p[eos]),"Nonfinite EOS logit");candidates.emplace_back(p[eos],-1);
            std::partial_sort(candidates.begin(),candidates.begin()+250,candidates.end(),[](const auto& a,const auto& b){return a.first>b.first||(a.first==b.first&&a.second<b.second);});
            std::vector<double> weights(250);for(int k=0;k<250;++k)weights[k]=std::exp((double(candidates[k].first)-candidates[0].first)/.95);
            std::discrete_distribution<int> sample(weights.begin(),weights.end());code=candidates[sample(random_)].second;if(code<0)return {};
        }
        previous_=code;codes_.push_back(code);std::vector<float> condition(small_?1024:2560);
        if(small_&&!smallHidden_){table_->row(code,1024,condition.data());}
        else{logits_=model_->forward({code+(small_?151830:1)},false); // consume-aligned hidden, as in supplied 4B specification
            LevoHiddenResearch::copySingleStepInto(*model_,condition.data(),width_,LevoHiddenResearch::SingleStepLayout::Channels1x1);}
        require(logits_.get()&&logits_->readMap<float>(),"D1 decode failed");return condition;
    }
    size_t kvPositions()const{return model_->getCurrentHistory();}
    const std::vector<int>& codes()const{return codes_;}
};
inline void little(std::ofstream& out,uint32_t value,int bytes){for(int i=0;i<bytes;++i)out.put(char((value>>(8*i))&255));}
inline void header(std::ofstream& out,uint32_t samples){out.seekp(0);out.write("RIFF",4);little(out,36+samples*4,4);out.write("WAVEfmt ",8);little(out,16,4);little(out,1,2);little(out,2,2);little(out,48000,4);little(out,192000,4);little(out,4,2);little(out,16,2);out.write("data",4);little(out,samples*4,4);}
inline void writeChunk(const std::string& path,const std::vector<float>& pcm,int count){
    std::string temp=path+".pending";std::ofstream out(temp,std::ios::binary);header(out,count);
    for(int t=0;t<count;++t)for(int c=0;c<2;++c){auto v=pcm[size_t(c)*491520+t];require(std::isfinite(v),"Invalid PCM");int16_t q=int16_t(std::clamp(v,-1.f,1.f)*32767);little(out,uint16_t(q),2);}
    out.close();require(out.good(),"Chunk file write failed");std::filesystem::rename(temp,path);
}
inline Json run(const std::string& appRoot,int choice,int requested,bool transport=false,const std::string& lyrics="",const std::string& npy="",const Json& options=Json::object()){
    require(requested>=25&&requested<=1500,"Experiment duration is1–60 seconds");
    const std::string smallCondition=options.value("smallCondition",std::string("token"));
    require(smallCondition=="token"||(choice==0&&smallCondition=="hidden-consume"),"Unknown condition experiment");
    const bool resident=options.value("resident",choice==0),cfg=options.value("cfg",choice!=0);
    const bool prefetch=options.value("prefetch",false);require(!prefetch||(resident&&choice==0),"Token prefetch requires resident small-model sessions");
    const std::string croppedLm=options.value("croppedLm",std::string());
    const int lmThreads=options.value("lmThreads",4),ditThreads=options.value("ditThreads",6),vaeThreads=options.value("vaeThreads",6);
    for(int n:{lmThreads,ditThreads,vaeThreads})require(n>=2&&n<=8,"Threads outside2–8 experiment range");
    const int steps=options.value("ditSteps",2),batch=!cfg&&options.value("singleBranch",false)?1:2;
    require(steps==2||steps==4||steps==7,"Unsupported step experiment");
    std::string id=std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    const std::string work=appRoot+"/stream-runs/"+id,root=appRoot+"/model-bundles/"+modelId(choice);std::filesystem::create_directories(work);
    {std::lock_guard<std::mutex> lock(stateMutex);state={{"runId",id},{"busy",true},{"status","generating"},{"phase","loading"},{"chunks",Json::array()},
       {"generatedSamples",0},{"requestedFrames",requested},{"conditionFrames",0},{"model",choice},{"conditionMode",choice==0?smallCondition:"hidden-projector"},{"transportOnly",transport},{"diagnosticOnly",lyrics.empty()||transport||smallCondition!="token"},{"musicQualityClaimed",false},
       {"resident",resident},{"tokenPrefetch",prefetch},{"threads",{{"lm",lmThreads},{"dit",ditThreads},{"vae",vaeThreads}}},{"croppedLm",croppedLm},{"ditSteps",steps},{"cfgEnabled",cfg},{"ditBatch",batch},{"unconditionalComputeRemoved",batch==1},{"chunkTimings",Json::array()}};}
    Trace trace;double lmTime=0,muTime=0,ditTime=0,vaeTime=0,ditSetup=0,vaeSetup=0;std::vector<float> conditions;std::unique_ptr<Prior> prior;
    std::unique_ptr<Graph> keptDit,keptVae;
    struct Fetched {std::vector<float> values;int count=0;bool eos=false;double elapsed=0;};std::future<Fetched> pending;
    size_t observedKv=0;
    std::vector<float> last,tail;std::mt19937 rng(2307);std::normal_distribution<float> normal;
    try{
        if(!transport){require(inspect(appRoot,choice)["ready"],"Import matching model bundle first");check();auto tick=Clock::now();prior=std::make_unique<Prior>(root,work,choice==0,lyrics,npy,smallCondition=="hidden-consume",croppedLm,lmThreads);lmTime+=seconds(tick);}
        const int hidden=choice==0?1024:2560;int produced=0,index=0;size_t emitted=0;
        std::vector<float> masks,mean,stddev;if(!transport){masks=read(root+"/aux/mask.f32",144);mean=read(root+"/aux/mean.f32",64);stddev=read(root+"/aux/std.f32",64);}
        int actual=requested;bool eos=false;
        for(int start=0;start<actual;start+=192,++index){
            const auto chunkTick=Clock::now();const double previousLm=lmTime,previousDit=ditTime,previousVae=vaeTime,previousDitSetup=ditSetup,previousVaeSetup=vaeSetup;
            if(pending.valid()){auto data=pending.get();conditions.insert(conditions.end(),data.values.begin(),data.values.end());produced+=data.count;lmTime+=data.elapsed;if(data.eos){eos=true;actual=produced;}patch({{"conditionFrames",produced},{"lmSeconds",lmTime}});}
            const int wanted=start+std::min(256,actual-start);
            patch({{"phase","decode"},{"chunk",index+1}});
            while(produced<wanted){check();if(transport){std::this_thread::sleep_for(std::chrono::milliseconds(2));++produced;}
                else{auto tick=Clock::now();auto h=prior->next(produced);lmTime+=seconds(tick);if(h.empty()){eos=true;actual=produced;break;}conditions.insert(conditions.end(),h.begin(),h.end());++produced;}
                if(produced%16==0||produced==wanted)patch({{"conditionFrames",produced},{"lmSeconds",lmTime}});
            }
            const int valid=std::min(256,actual-start);if(valid<=0)break;const bool final=start+valid==actual;
            const bool tailOnly=eos&&index&&valid==64;
            std::vector<float> pcm(2*491520);
            if(tailOnly){for(int c=0;c<2;++c)std::copy_n(tail.data()+c*64*1920,64*1920,pcm.data()+c*491520);}
            else if(transport){for(int c=0;c<2;++c)for(int i=0;i<491520;++i)pcm[size_t(c)*491520+i]=.02f*std::sin(2*3.141592653589793*220*(start*1920+i)/48000.);}
            else {
                std::vector<float> mu(256*1024);
                if(choice==0){for(int row=0;row<256;++row)std::copy_n(conditions.data()+size_t((start+row)%produced)*1024,1024,mu.data()+row*1024);}
                else {patch({{"phase","projection"}});auto tick=Clock::now();Graph projector(root+"/mu/mu.mnn",false,4);std::vector<float> input(2048*2560);
                    for(int row=0;row<256;++row)std::copy_n(conditions.data()+size_t((start+row)%produced)*2560,2560,input.data()+row*2560);
                    auto output=projector.run({{"",std::move(input)}},2048*1024);std::copy_n(output.data(),256*1024,mu.data());muTime+=seconds(tick);}
                std::vector<float> context(256*64,0),noise(256*64),x;
                if(!last.empty())std::copy_n(last.data()+192*64,64*64,context.data());for(auto& v:noise)v=normal(rng);x=noise;
                observedKv=prior->kvPositions();
                if(prefetch){
                    // Finish all graph creation/vector-sine hook changes before
                    // another thread uses the prior. Each model has one owner.
                    if(!keptDit){auto tick=Clock::now();keptDit=std::make_unique<Graph>(root+"/dit/dit.mnn",false,ditThreads,false,batch);double elapsed=seconds(tick);ditSetup+=elapsed;ditTime+=elapsed;}
                    if(!keptVae){auto tick=Clock::now();keptVae=std::make_unique<Graph>(root+"/vae/vae.mnn",true,vaeThreads,true);double elapsed=seconds(tick);vaeSetup+=elapsed;vaeTime+=elapsed;}
                    const int nextStart=start+192,nextWanted=nextStart+std::min(256,actual-nextStart),from=produced;
                    if(!final&&!eos&&nextWanted>from){pending=std::async(std::launch::async,[&,from,nextWanted]{
                        Fetched data;data.values.reserve(size_t(nextWanted-from)*1024);auto tick=Clock::now();
                        for(int at=from;at<nextWanted;++at){check();auto value=prior->next(at);if(value.empty()){data.eos=true;break;}data.values.insert(data.values.end(),value.begin(),value.end());++data.count;}
                        data.elapsed=seconds(tick);return data;});}
                }
                {patch({{"phase","dit"}});auto tick=Clock::now();if(!keptDit){auto setup=Clock::now();keptDit=std::make_unique<Graph>(root+"/dit/dit.mnn",false,ditThreads,false,batch);ditSetup+=seconds(setup);}check();
                    for(int step=0;step<steps;++step){check();patch({{"step",step+1},{"steps",steps}});float t=float(step)/steps;
                        if(index)for(int j=0;j<64*64;++j)x[j]=(1-(1-1e-4f)*t)*noise[j]+t*context[j];
                        std::vector<float> input(batch*256*1200,0),attn(batch*256*256,0);
                        for(int b=0;b<batch;++b)for(int row=0;row<256;++row){size_t off=size_t(b*256+row)*1200;int mask=row>=valid?0:(index&&row<64?1:2);
                            std::copy_n(masks.data()+mask*48,48,input.data()+off);std::copy_n(context.data()+row*64,64,input.data()+off+48);
                            if(!cfg||b)std::copy_n(mu.data()+row*1024,1024,input.data()+off+112);std::copy_n(x.data()+row*64,64,input.data()+off+1136);
                            if(row<valid)std::fill_n(attn.data()+size_t(b*256+row)*256,valid,1.f);}
                        auto v=keptDit->run({{"inputs",std::move(input)},{"timestep",std::vector<float>(batch,t)},{"attention_mask",std::move(attn)}},batch*256*64);
                        for(size_t j=0;j<x.size();++j)x[j]+=(cfg?v[j]+1.5f*(v[x.size()+j]-v[j]):v[j])/steps;peak(x);}
                    if(!resident)keptDit.reset();ditTime+=seconds(tick);}
                if(index)std::copy_n(context.data(),64*64,x.data());last=x;check();
                std::vector<float> latent(64*256);for(int c=0;c<64;++c)for(int t=0;t<256;++t)latent[c*256+t]=x[t*64+c]*stddev[c]+mean[c];
                {patch({{"phase","vae"}});auto tick=Clock::now();if(!keptVae){auto setup=Clock::now();keptVae=std::make_unique<Graph>(root+"/vae/vae.mnn",true,vaeThreads,true);vaeSetup+=seconds(setup);}
                    pcm=keptVae->run({{"",std::move(latent)}},2*491520);if(!resident)keptVae.reset();vaeTime+=seconds(tick);}
            }
            check();if(index)for(int c=0;c<2;++c)for(int i=0;i<64*1920;++i){float r=float(i)/(64*1920-1);pcm[size_t(c)*491520+i]=tail[size_t(c)*64*1920+i]*(1-r)+pcm[size_t(c)*491520+i]*r;}
            const int count=(valid-(final?0:64))*1920;const std::string path=work+"/chunk-"+std::to_string(index)+".wav";
            writeChunk(path,pcm,count);check();if(!final){tail.resize(2*64*1920);for(int c=0;c<2;++c)std::copy_n(pcm.data()+size_t(c)*491520+192*1920,64*1920,tail.data()+size_t(c)*64*1920);}
            {std::lock_guard<std::mutex> lock(stateMutex);state["chunks"].push_back({{"path",path},{"startSample",emitted},{"samples",count},{"index",index}});emitted+=count;state["generatedSamples"]=emitted;state["phase"]="published";
                state["chunkTimings"].push_back({{"index",index},{"lmSeconds",lmTime-previousLm},{"ditSeconds",ditTime-previousDit},{"ditSetupSeconds",ditSetup-previousDitSetup},{"vaeSeconds",vaeTime-previousVae},{"vaeSetupSeconds",vaeSetup-previousVaeSetup},{"wallSeconds",seconds(chunkTick)},{"rssMiB",::resident()/1024.}});}
            trace.sample();patch({{"elapsedSeconds",seconds(trace.start)},{"rssPeakMiB",trace.maxRss.load()/1024.},{"lmSeconds",lmTime},{"muSeconds",muTime},{"ditSeconds",ditTime},{"vaeSeconds",vaeTime},
                {"kvPositions",pending.valid()?observedKv:(prior?prior->kvPositions():0)},{"firstChunkSeconds",index==0?seconds(trace.start):snapshot().value("firstChunkSeconds",0.)}});
            if(final)break;
        }
        require(snapshot()["generatedSamples"].get<size_t>()==size_t(actual)*1920,"Published duration mismatch");patch({{"status","completed"},{"actualFrames",actual},{"endedWithEos",eos}});
    }catch(const std::exception& e){patch({{"status",cancelled?"stopped":"error"},{"error",e.what()}});}
    if(pending.valid()){try{pending.get();}catch(...){} }
    if(prior){std::ofstream(work+"/codes.json")<<Json(prior->codes()).dump();}
    keptVae.reset();keptDit.reset();prior.reset();trace.sample();patch({{"busy",false},{"elapsedSeconds",seconds(trace.start)},{"rssPeakMiB",trace.maxRss.load()/1024.},{"ditSetupSeconds",ditSetup},{"vaeSetupSeconds",vaeSetup},
        {"minimumAvailableMiB",trace.minAvailable.load()/1024.},{"lmSeconds",lmTime},{"muSeconds",muTime},{"ditSeconds",ditTime},{"vaeSeconds",vaeTime}});
    auto result=snapshot();std::ofstream(work+"/result.json")<<result.dump(2);return result;
}
}
