#include "levo/RuntimeSupport.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include "MNN_generated.h"
using namespace MNN::Express;
extern "C" uint64_t LlmOperatorCandidateCalls() __attribute__((weak));
#ifndef LLM_OPERATOR_VARIANT_ID
#define LLM_OPERATOR_VARIANT_ID "baseline"
#endif
int main(int argc,char** argv){
    if(argc!=4){std::cerr<<"fixture-dir report.json threads\n";return 2;}
    if(std::filesystem::exists(argv[2])){std::cerr<<"Preserve previous measurement\n";return 2;}
    Trace trace;Json report={{"schemaVersion",1},{"status","error"},{"variantId",LLM_OPERATOR_VARIANT_ID},{"rows",Json::array()},
      {"scope","Isolated existing MNN MUL_SILU operator, actual first-block MLP activations. Not new graph fusion or whole-LLM speedup."}};
    try{
        const int threads=std::stoi(argv[3]);require(threads==4,"Fixed4threads protocol");trace.gate(1024);
        MNN::BackendConfig config;config.precision=MNN::BackendConfig::Precision_High;config.memory=MNN::BackendConfig::Memory_Low;config.power=MNN::BackendConfig::Power_High;
        Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,config,threads);
        auto fixtures=readJson(std::string(argv[1])+"/fixtures.json");
        for(const std::string phase:{"prefill","decode"}){
            auto f=fixtures["phases"][phase];const int count=f["input0"]["elements"];auto a=read(std::string(argv[1])+"/"+f["input0"]["file"].get<std::string>(),count),b=read(std::string(argv[1])+"/"+f["input1"]["file"].get<std::string>(),count),reference=read(std::string(argv[1])+"/"+f["output"]["file"].get<std::string>(),count);
            std::vector<int> shape=f["input0"]["shape"].get<std::vector<int>>();require(shape==f["input1"]["shape"].get<std::vector<int>>()&&shape==f["output"]["shape"].get<std::vector<int>>(),"Elementwise shapes differ");
            auto x=_Input(shape,NCHW,halide_type_of<float>()),z=_Input(shape,NCHW,halide_type_of<float>());
            std::unique_ptr<MNN::OpT> op(new MNN::OpT);op->type=MNN::OpType_BinaryOp;op->main.type=MNN::OpParameter_BinaryOp;op->main.value=new MNN::BinaryOpT;op->main.AsBinaryOp()->opType=MNN::BinaryOpOperation_MUL_SILU;
            auto y=Variable::create(Expr::create(std::move(op),{x,z}));std::vector<double> times;double first=0;const float* result=nullptr;
            for(int rep=0;rep<28;++rep){
                std::memcpy(x->writeMap<float>(),a.data(),count*4);std::memcpy(z->writeMap<float>(),b.data(),count*4);auto tick=Clock::now();result=y->readMap<float>();const double ms=seconds(tick)*1000;
                require(result&&y->getInfo()->size==count,"MUL_SILU execution failed");if(rep==0)first=ms;else if(rep>=3)times.push_back(ms);
            }
            std::vector<float> actual(result,result+count);peak(actual);double l2=relative(actual,reference);require(l2<1e-5,"Actual activation regression failed");std::sort(times.begin(),times.end());
            report["rows"].push_back({{"phase",phase},{"elements",count},{"firstMs",first},{"warmMedianMs",times[times.size()/2]},{"warmMinMs",times.front()},{"warmMaxMs",times.back()},
                {"warmSamplesMs",times},{"relativeL2",l2},{"checkedElements",count},{"calls",28}});
        }
        report["candidateCalls"]=LlmOperatorCandidateCalls?LlmOperatorCandidateCalls():0;report["status"]="completed";
    }catch(const std::exception& e){report["error"]=e.what();}
    trace.sample();report["rssPeakMiB"]=trace.maxRss.load()/1024.;report["threads"]=4;report["precision"]="high";report["elapsedSeconds"]=seconds(trace.start);
    std::ofstream out(argv[2]);out<<report.dump(2);std::cout<<report["status"]<<std::endl;return report["status"]=="completed"?0:1;
}
