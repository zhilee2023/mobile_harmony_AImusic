// Bounded whole-decoder activation sensitivity; not an INT4 speed claim.
#include "../runtime/include/levo/experimental/StreamLab.hpp"
#include "../runtime/include/levo/kernels/SnakeFusionGraph.hpp"
using namespace LevoStreamLab;
namespace {
struct Trial{const char* id;int bits;bool rotate;size_t minimumArea;};
Json error(const std::vector<float>& output,const std::vector<float>& reference){
    double maxAbs=0;size_t clipping=0;for(size_t i=0;i<output.size();++i){maxAbs=std::max(maxAbs,std::abs(double(output[i])-reference[i]));clipping+=std::abs(output[i])>1;}
    const double l2=relative(output,reference);
    return {{"relativeL2",l2},{"maxAbs",maxAbs},{"snrDb",l2>0?-20*std::log10(l2):200.},{"outsideUnitPcm",clipping},{"accepted",l2<=.01&&maxAbs<=.01&&clipping==0}};
}
void rotationTest(){
    alignas(16) float input[32],output[32];float32x4_t v[8];for(int i=0;i<32;++i)input[i]=float(std::sin(i*.7)*3);
    for(int i=0;i<8;++i)v[i]=vld1q_f32(input+4*i);LevoVaeActivationResearch::hadamard(v);LevoVaeActivationResearch::hadamard(v);
    for(int i=0;i<8;++i)vst1q_f32(output+4*i,vmulq_n_f32(v[i],1.f/32));
    for(int i=0;i<32;++i)require(std::abs(input[i]-output[i])<2e-6,"Hadamard inverse failed");
}
std::vector<float> forward(Graph& decoder,const std::vector<float>& latent,const Trial& trial,Json& result){
    LevoVaeActivationResearch::bits=trial.bits;LevoVaeActivationResearch::rotate=trial.rotate;LevoVaeActivationResearch::minimumArea=trial.minimumArea;
    LevoVaeActivationResearch::calls=0;LevoVaeActivationResearch::elements=0;auto tick=Clock::now();
    auto pcm=decoder.run({{"",latent}},2*491520);
    result={{"id",trial.id},{"activationBitsSimulated",trial.bits},{"signedHadamard32",trial.rotate},{"minimumTimeArea",trial.minimumArea},
            {"diagnosticSeconds",seconds(tick)},{"quantizedSnakeNodes",LevoVaeActivationResearch::calls.load()},{"quantizedElements",LevoVaeActivationResearch::elements.load()}};
    return pcm;
}
}
int main(int argc,char** argv){
    if(argc!=5)return 2;const std::string root=argv[1],first=argv[2],tail=argv[3],out=argv[4];
    Json report={{"success",false},{"scope","Real W8/FP16 decoder, fused Snake+dynamic group32 activation QDQ, optional signed Hadamard/inverse. FP16 outputs and original weights; not packed A4 or int4/int8 convolution speed/deployment"},
                 {"maximumForwards",9},{"l2Budget",.01},{"maxAbsBudget",.01},{"deploymentEnabled",false}};Trace trace;
    try{
        require(!std::filesystem::exists(out),"Preserve previous quantization experiments");std::filesystem::create_directories(out);rotationTest();report["hadamardInversePassed"]=true;
        auto latent=read(first,64*256);Graph decoder(root+"/vae/vae.mnn",true,6,true,0,true);Json baselineRow;
        const Trial baseline{"fp16-fused-reference",0,false,0};auto reference=forward(decoder,latent,baseline,baselineRow);report["baseline"]=baselineRow;
        const std::vector<Trial> trials={{"a8-all",8,false,0},{"a8-rot32-all",8,true,0},{"a8-hot",8,false,122880},{"a8-rot32-hot",8,true,122880},{"a4-hot",4,false,122880},{"a4-rot32-hot",4,true,122880}};
        report["trials"]=Json::array();int best=-1;double lowestError=1e30;
        for(size_t i=0;i<trials.size();++i){check();require(seconds(trace.start)<150,"Bounded research time reached; remaining trials not run");Json row;auto pcm=forward(decoder,latent,trials[i],row);auto metric=error(pcm,reference);row["pcm"]=metric;
            require(row["quantizedSnakeNodes"].get<int>()>0,"No supported activation node was quantized");
            if(metric["accepted"]==true&&metric["relativeL2"].get<double>()<lowestError){best=int(i);lowestError=metric["relativeL2"].get<double>();}
            report["trials"].push_back(row);std::ofstream(out+"/result.json")<<report.dump(2);std::cout<<"VAE_ACTIVATION_TRIAL "<<row.dump()<<std::endl;
        }
        if(best>=0){require(seconds(trace.start)<150,"Bounded research time reached before held-out confirmation");auto heldOut=read(tail,64*256);Json row,base;auto expected=forward(decoder,heldOut,baseline,base);auto actual=forward(decoder,heldOut,trials[best],row);
            row["pcm"]=error(actual,expected);report["heldOutTail"]=row;report["candidatePassedBothFixtures"]=row["pcm"]["accepted"];
        }else{report["candidatePassedBothFixtures"]=false;report["conclusion"]="All six bounded candidates exceeded the declared budget; no deployment";}
        report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["minimumAvailableMiB"]=trace.minAvailable.load()/1024.;
    if(std::filesystem::is_directory(out))std::ofstream(out+"/result.json")<<report.dump(2);std::cout<<"VAE_ACTIVATION_DONE "<<report.value("success",false)<<std::endl;return report["success"]==true?0:1;
}
