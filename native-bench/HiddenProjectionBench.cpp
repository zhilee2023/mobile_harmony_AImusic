// Framewise affine projection only. Fixture weights are SYNTHETIC, not a
// trained private conditioner. Batching is not valid for arbitrary temporal
// conditioners, normalization across frames, or recurrent state.
#include "../runtime/include/levo/RuntimeSupport.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "../runtime/include/levo/experimental/FramewiseAffine.hpp"

using namespace MNN::Express;

int main(int argc,char** argv) {
    if(argc!=4){std::cerr<<"hidden256x1024.f32 threads report.json\n";return 2;}
    if(std::filesystem::exists(argv[3])){std::cerr<<"Keep previous report\n";return 2;}
    Trace trace;Json report={{"success",false},{"scope","Isolated MNN CPU FP32 affine fixture; real student hidden, synthetic projector weights; not music or full-chain RTF"}};
    try {
        const int threads=std::stoi(argv[2]),width=1024,outputWidth=1024;
        require(threads==4||threads==6,"Bounded threads");
        MNN::BackendConfig config;config.precision=MNN::BackendConfig::Precision_High;
        config.memory=MNN::BackendConfig::Memory_Low;config.power=MNN::BackendConfig::Power_High;
        Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,config,threads);
        const auto hidden=read(argv[1],256*width);peak(hidden);
        std::mt19937 rng(2307);std::uniform_real_distribution<float> random(-.02f,.02f);
        std::vector<float> values(width*outputWidth),biasValues(outputWidth);
        for(float& value:values)value=random(rng);
        for(float& value:biasValues)value=random(rng);
        Json rows=Json::array();
        for(int frames:{64,256}) {
            LevoHiddenResearch::FramewiseAffine single(1,width,outputWidth,values,biasValues),batch(frames,width,outputWidth,values,biasValues);
            std::vector<float> reference(frames*outputWidth),candidate(reference.size());
            auto runSingle=[&](){
                for(int i=0;i<frames;++i){
                    single.runInto(hidden.data()+i*width,width,reference.data()+i*outputWidth,outputWidth);
                }
            };
            auto runBatch=[&](){
                batch.runInto(hidden.data(),frames*width,candidate.data(),frames*outputWidth);
            };
            auto tick=Clock::now();runSingle();const double firstSingle=seconds(tick);
            tick=Clock::now();runBatch();const double firstBatch=seconds(tick);
            Json paired=Json::array();
            for(int repeat=0;repeat<5;++repeat) {
                double a,b;
                if(repeat%2==0){tick=Clock::now();runSingle();a=seconds(tick);tick=Clock::now();runBatch();b=seconds(tick);}
                else {tick=Clock::now();runBatch();b=seconds(tick);tick=Clock::now();runSingle();a=seconds(tick);}
                peak(reference);peak(candidate);double error=relative(candidate,reference);
                require(error<1e-5,"Batching changed projection beyond FP32 tolerance");
                paired.push_back({{"repeat",repeat+1},{"perFrameSeconds",a},{"batchSeconds",b},{"relativeL2",error}});
            }
            rows.push_back({{"frames",frames},{"inputWidth",width},{"outputWidth",outputWidth},
                {"firstPerFrameSeconds",firstSingle},{"firstBatchSeconds",firstBatch},{"pairs",paired}});
        }
        report["requestedThreads"]=threads;report["precision"]="high";report["syntheticWeightSeed"]=2307;
        report["weightBits"]=32;report["rows"]=rows;report["success"]=true;
    }catch(const std::exception& error){report["error"]=error.what();}
    trace.sample();report["totalSeconds"]=seconds(trace.start);report["sampledPeakRssKiB"]=trace.maxRss.load();
    report["minimumSystemAvailableKiB"]=trace.minAvailable.load();
    {std::ofstream output(argv[3]);output<<report.dump(2);}
    std::cout<<"LEVO_HIDDEN_PROJECTION_REPORT "<<report.dump()<<std::endl;
    return report["success"].get<bool>()?0:1;
}
