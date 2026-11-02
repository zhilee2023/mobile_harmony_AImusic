// Actual pretrained-LM codes, rendered on the connected Harmony phone.
// Source metadata distinguishes PC-generated codes from whole-phone runs.
#define LEVO_CHAIN_LIBRARY
#include "ChainBench.cpp"

static void little(std::ofstream& out,uint32_t value,int bytes){
    for(int i=0;i<bytes;++i)out.put(char((value>>(8*i))&255));
}

int LevoPublicCodecMain(int argc,char** argv,Json* captured=nullptr){
    if(argc!=5){std::cerr<<"codecRoot codes.json steps output.wav\n";return 2;}
    Trace trace;Json report={{"success",false},{"lmDevice","Windows CPU"},{"codecDevice","Harmony phone CPU"},
        {"trainedLmTokens",true},{"randomMusicHead",false},{"wholePhoneChain",false}};
    try{
        std::string root=argv[1];int steps=std::stoi(argv[3]);require(steps>=2&&steps<=20,"Bounded steps");
        auto source=readJson(argv[2]);require(source["source"]=="trained public LeVo v2-large primary Mixed LM","Wrong token source");
        report["lmDevice"]=source.value("lmDevice",std::string("Windows CPU"));
        report["wholePhoneChain"]=source.value("lmDevice",std::string())=="Harmony phone CPU";
        auto codes=source["codes"].get<std::vector<int>>();require(!codes.empty()&&codes.size()<=1500,"Bounded audio");
        for(int id:codes)require(id>=0&&id<16384,"Invalid Mixed ID");
        report["ditSteps"]=steps;report["lmGuidance"]=source["lmGuidance"];report["codecGuidance"]=1.5;
        report["ditWeightBits"]=8;report["vaeWeightBits"]=8;report["lmWeightBits"]=4;
        trace.gate(1536);trace.mark("real-public-codes-start");
        auto tick=Clock::now();Session dit(root+"/dit/full-w8.mnn",false,6);report["ditSetupSeconds"]=seconds(tick);
        trace.mark("real-dit-loaded");trace.gate(1024);
        tick=Clock::now();std::unique_ptr<Session> vae;
        {LevoVaeSin::Scope sinScope(true);vae=std::make_unique<Session>(root+"/vae/vae-256-w8.mnn",true,6);}
        report["vaeSetupSeconds"]=seconds(tick);report["vaeSinHookDuringBuild"]=true;
        trace.mark("real-vae-loaded");
        auto book=read(root+"/aux/codebook.f32",16384*32),weight=read(root+"/aux/weight.f32",1024*32),bias=read(root+"/aux/bias.f32",1024);
        auto masks=read(root+"/aux/mask.f32",3*48),mean=read(root+"/aux/mean.f32",64),stddev=read(root+"/aux/std.f32",64);
        constexpr int frames=256,hop=192,contextFrames=64,pcmFrames=frames*1920,pcmOverlap=contextFrames*1920;
        std::mt19937 rng(2307);std::normal_distribution<float> normal;
        std::vector<float> last,tail;Json chunks=Json::array();double energy=0,maxPcm=0;size_t clipped=0,emitted=0;
        std::ofstream output(argv[4],std::ios::binary);require(output.good(),"Cannot write WAV");
        const uint32_t totalSamples=codes.size()*1920;
        output.write("RIFF",4);little(output,36+totalSamples*4,4);output.write("WAVEfmt ",8);little(output,16,4);
        little(output,1,2);little(output,2,2);little(output,48000,4);little(output,192000,4);little(output,4,2);little(output,16,2);
        output.write("data",4);little(output,totalSamples*4,4);
        int index=0;
        for(size_t start=0;;start+=hop,++index){
            trace.gate(800);int valid=std::min(size_t(frames),codes.size()-start);bool final=start+valid>=codes.size();
            std::vector<float> mu(frames*1024),context(frames*64,0),noise(frames*64),x;
            for(int t=0;t<valid;++t)for(int c=0;c<1024;++c){float sum=bias[c];for(int k=0;k<32;++k)sum+=book[codes[start+t]*32+k]*weight[c*32+k];mu[t*1024+c]=sum;}
            if(!last.empty())std::copy_n(last.data()+(frames-contextFrames)*64,contextFrames*64,context.data());
            for(auto& value:noise)value=normal(rng);x=noise;
            double ditTime=0;
            for(int step=0;step<steps;++step){
                trace.gate(800);float t=float(step)/steps;
                if(index)for(int i=0;i<contextFrames*64;++i)x[i]=(1-(1-1e-4f)*t)*noise[i]+t*context[i];
                std::vector<float> input(2*frames*1200,0),attn(2*frames*frames,0);
                for(int b=0;b<2;++b)for(int row=0;row<frames;++row){
                    int id=row>=valid?0:(index&&row<contextFrames?1:2);size_t off=(b*frames+row)*1200;
                    std::copy_n(masks.data()+id*48,48,input.data()+off);
                    std::copy_n(context.data()+row*64,64,input.data()+off+48);
                    if(b)std::copy_n(mu.data()+row*1024,1024,input.data()+off+112);
                    std::copy_n(x.data()+row*64,64,input.data()+off+1136);
                    if(row<valid)std::fill_n(attn.data()+(b*frames+row)*frames,valid,1.f);
                }
                tick=Clock::now();auto velocity=dit.run({{"inputs",std::move(input)},{"timestep",{t,t}},{"attention_mask",std::move(attn)}},"velocity",2*frames*64);
                ditTime+=seconds(tick);for(size_t j=0;j<x.size();++j)x[j]+=(velocity[j]+1.5f*(velocity[x.size()+j]-velocity[j]))/steps;
                peak(x);trace.mark("chunk-"+std::to_string(index+1)+"-dit-"+std::to_string(step+1));
            }
            if(index)std::copy_n(context.data(),contextFrames*64,x.data());last=x;
            std::vector<float> latent(frames*64);for(int c=0;c<64;++c)for(int t=0;t<frames;++t)latent[c*frames+t]=x[t*64+c]*stddev[c]+mean[c];
            tick=Clock::now();auto pcm=vae->run({{"latent",std::move(latent)}},"pcm",2*pcmFrames);double vaeTime=seconds(tick);
            if(index)for(int c=0;c<2;++c)for(int i=0;i<pcmOverlap;++i){float r=float(i)/(pcmOverlap-1);pcm[c*pcmFrames+i]=tail[c*pcmOverlap+i]*(1-r)+pcm[c*pcmFrames+i]*r;}
            int count=valid*1920-(final?0:pcmOverlap);
            for(int t=0;t<count;++t)for(int c=0;c<2;++c){float value=pcm[c*pcmFrames+t];energy+=double(value)*value;maxPcm=std::max(maxPcm,std::abs(double(value)));clipped+=std::abs(value)>1;
                int16_t sample=int16_t(std::max(-1.f,std::min(1.f,value))*32767);little(output,uint16_t(sample),2);}
            emitted+=count;
            if(!final){tail.resize(2*pcmOverlap);for(int c=0;c<2;++c)std::copy_n(pcm.data()+c*pcmFrames+valid*1920-pcmOverlap,pcmOverlap,tail.data()+c*pcmOverlap);}
            chunks.push_back({{"chunk",index+1},{"validFrames",valid},{"ditSeconds",ditTime},{"vaeSeconds",vaeTime},{"pcmPeak",peak(pcm)}});
            output.flush();trace.mark("chunk-"+std::to_string(index+1)+"-wav-written");
            if(final)break;
        }
        require(emitted==totalSamples&&output.good(),"WAV duration/write failure");
        report["chunks"]=chunks;report["durationSeconds"]=double(totalSamples)/48000;report["pcmRms"]=std::sqrt(energy/(2*emitted));
        report["pcmPeak"]=maxPcm;report["clippedSampleFraction"]=double(clipped)/(2*emitted);
        report["codecSeconds"]=seconds(trace.start);report["codecOnlyRtf"]=seconds(trace.start)/(double(totalSamples)/48000);
        report["vaeNeonSinCalls"]=LevoVaeSin::calls.load();report["audioFileGenerated"]=true;report["success"]=true;
    }catch(const std::exception& error){report["error"]=error.what();}
    trace.sample();report["sampledPeakRssKiB"]=trace.maxRss.load();report["minimumSystemAvailableKiB"]=trace.minAvailable.load();
    report["totalSeconds"]=seconds(trace.start);report["stages"]=trace.stages;
    if(captured)*captured=report;
    std::cout<<"LEVO_PUBLIC_CODEC_REPORT "<<report.dump()<<std::endl;return report["success"].get<bool>()?0:1;
}
#ifndef LEVO_PUBLIC_CODEC_LIBRARY
int main(int argc,char** argv){return LevoPublicCodecMain(argc,argv);}
#endif
