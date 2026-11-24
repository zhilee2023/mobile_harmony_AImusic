// Independent FP32 operator/nominal-bandwidth probes, not quantized LM kernels.
#include "levo/RuntimeSupport.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "MNN_generated.h"
#ifndef LLM_OPERATOR_MNN_SHA256
#define LLM_OPERATOR_MNN_SHA256 "not-recorded"
#endif
using namespace MNN::Express;
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
static VARP rms(VARP x,int width){
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);op->type=MNN::OpType_LayerNorm;
    op->main.type=MNN::OpParameter_LayerNorm;op->main.value=new MNN::LayerNormT;
    auto p=op->main.AsLayerNorm();p->axis={-1};p->epsilon=1e-6f;p->useRMSNorm=true;p->gamma.assign(width,1.f);
    return Variable::create(Expr::create(std::move(op),{x}));
}
static Json probe(const std::string& kind,int rows,int width,int outWidth,const std::vector<float>& prefix){
    const auto tick=Clock::now();const size_t count=size_t(rows)*width;
    std::vector<float> a(count),b;for(size_t i=0;i<count;++i)a[i]=prefix[i%prefix.size()];
    auto x=_Input({rows,width},NCHW,halide_type_of<float>());VARP y;
    uint64_t flops=0,logical=0;size_t outputCount=count;
    if(kind=="MatMul"){
        b.resize(size_t(width)*outWidth);std::mt19937 rng(42);std::uniform_real_distribution<float> d(-.02f,.02f);for(auto& v:b)v=d(rng);
        auto w=_Const(b.data(),{width,outWidth},NCHW,halide_type_of<float>());y=_MatMul(x,w);outputCount=size_t(rows)*outWidth;
        flops=2ULL*rows*width*outWidth;logical=4ULL*(a.size()+b.size()+outputCount);
    }else if(kind=="RMSNorm"){y=rms(x,width);logical=4ULL*(2*count+width);}
    else if(kind=="SiLU"){y=_Silu(x);logical=8ULL*count;}
    else if(kind=="ResidualAdd"){b=a;std::reverse(b.begin(),b.end());y=_Add(x,_Const(b.data(),{rows,width},NCHW,halide_type_of<float>()));logical=12ULL*count;}
    else if(kind=="GateMultiply"){b=a;std::reverse(b.begin(),b.end());y=_Multiply(x,_Const(b.data(),{rows,width},NCHW,halide_type_of<float>()));logical=12ULL*count;}
    else throw std::runtime_error("Unknown microbenchmark");
    const double construction=seconds(tick)*1000;std::vector<double> ms;double first=0;const float* result=nullptr;
    for(int repeat=0;repeat<8;++repeat){
        std::memcpy(x->writeMap<float>(),a.data(),a.size()*4);auto start=Clock::now();result=y->readMap<float>();const double t=seconds(start)*1000;
        require(result&&y->getInfo()->size==outputCount,"Microbenchmark shape/sync failed");if(repeat==0)first=t;else if(repeat>=3)ms.push_back(t);
    }
    double error=0,energy=0,maxAbs=0;size_t checked=0;
    auto check=[&](size_t index,double expected){require(std::isfinite(result[index]),"Nonfinite operator output");double diff=result[index]-expected;error+=diff*diff;energy+=expected*expected;maxAbs=std::max(maxAbs,std::abs(diff));++checked;};
    if(kind=="MatMul"){
        // Selected dense products are checked with double accumulation, not a full-matrix correctness claim.
        for(int r:rows==1?std::vector<int>{0}:std::vector<int>{0,rows-1})for(int c:{0,outWidth/3,outWidth/2,outWidth-1}){double v=0;for(int k=0;k<width;++k)v+=double(a[size_t(r)*width+k])*b[size_t(k)*outWidth+c];check(size_t(r)*outWidth+c,v);}
    }else for(int r=0;r<rows;++r){double inv=1;if(kind=="RMSNorm"){double e=0;for(int c=0;c<width;++c){double v=a[size_t(r)*width+c];e+=v*v;}inv=1/std::sqrt(e/width+1e-6);}
        for(int c=0;c<width;++c){size_t i=size_t(r)*width+c;double v=a[i];if(kind=="RMSNorm")v*=inv;else if(kind=="SiLU")v/=1+std::exp(-v);else if(kind=="ResidualAdd")v+=b[i];else if(kind=="GateMultiply")v*=b[i];check(i,v);}}
    const double l2=std::sqrt(error/std::max(energy,1e-30));require(l2<1e-4,"FP32 microbenchmark failed numerical gate");
    return {{"operator",kind},{"rows",rows},{"inputWidth",width},{"outputWidth",kind=="MatMul"?outWidth:width},{"constructionMs",construction},{"firstMs",first},{"warmMedianMs",median(ms)},{"warmSamplesMs",ms},
        {"relativeL2",l2},{"maxAbs",maxAbs},{"checkedElements",checked},{"outputElements",outputCount},{"validationScope",kind=="MatMul"?"4 or8 selected entries, double accumulation":"all output elements against scalar formula"},
        {"flops",flops},{"flopsScope",kind=="MatMul"?"2*M*K*N":"not assigned to mixed/transcendental work"},{"logicalBytes",logical},
        {"attainedFlopsPerSecond",flops?flops/(median(ms)/1000):0.},{"syntheticParameters",kind=="MatMul"||kind=="RMSNorm"},{"activationScope","Actual style-prefix values; not captured internal activation distributions"}};
}
int main(int argc,char** argv){
    if(argc!=4){std::cerr<<"case-root report.json threads\n";return 2;}
    if(std::filesystem::exists(argv[2])){std::cerr<<"Preserve previous calibration\n";return 2;}
    Trace trace;Json report={{"schemaVersion",1},{"success",false},{"scope","Isolated CLI MNN CPU FP32/high probes. Real prefix values, synthetic weights/unit gamma. Not exact trained W4/W8 kernels, not App latency or music RTF."},{"rows",Json::array()}};
    try{
        int threads=std::stoi(argv[3]);require(threads==4,"This calibration protocol is fixed to4threads");trace.gate(1024);
        MNN::BackendConfig config;config.precision=MNN::BackendConfig::Precision_High;config.memory=MNN::BackendConfig::Memory_Low;config.power=MNN::BackendConfig::Power_High;
        Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,config,threads);report["threads"]=threads;report["precision"]="high";report["mnnArchiveSha256"]=LLM_OPERATOR_MNN_SHA256;report["affinity"]="OS scheduler; no core pinning";
        for(bool large:{false,true}){
            const int hidden=large?2560:1024,ffn=large?9728:3072;auto prefix=read(std::string(argv[1])+"/cases/"+(large?"large":"small")+"/fixtures/prefix.f32",513*hidden);
            for(int rows:{1,513})for(const std::string kind:{"MatMul","RMSNorm","SiLU","ResidualAdd","GateMultiply"}){
                trace.gate(1024);auto value=probe(kind,rows,kind=="GateMultiply"?ffn:hidden,ffn,prefix);value["modelShape"]=large?"large":"small";value["phaseShape"]=rows==1?"decode":"prefill";report["rows"].push_back(value);
                std::cout<<kind<<" "<<(large?"large":"small")<<" "<<rows<<" "<<value["warmMedianMs"]<<" ms\n";
            }
        }
        {const size_t n=32ULL*1024*1024;std::vector<float> a(n,1.25f),b(n);std::vector<double> times;
            for(int r=0;r<6;++r){auto t=Clock::now();std::vector<std::thread> workers;for(int i=0;i<threads;++i)workers.emplace_back([&,i]{size_t first=n*i/threads,last=n*(i+1)/threads;std::memcpy(b.data()+first,a.data()+first,(last-first)*4);});for(auto& w:workers)w.join();if(r)times.push_back(seconds(t));}
            require(a==b,"Bandwidth copy verification failed");const double secondsMedian=median(times);
            report["bandwidth"]={{"bytesPerArray",n*4},{"arrays",2},{"threads",threads},{"medianSeconds",secondsMedian},{"nominalBytesPerSecond",double(n*8)/secondsMedian},
                {"scope","Read+write nominal bytes of128MiB copy, 4workers; includes thread launch/join; not memory-controller DRAM counters or chip peak"}};}
        report["success"]=true;
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["elapsedSeconds"]=seconds(trace.start);
    std::ofstream out(argv[2]);out<<report.dump(2);return report["success"].get<bool>()?0:1;
}
