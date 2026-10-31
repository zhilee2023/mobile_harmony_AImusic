// Debug-only execution audit, separate from performance measurements.
#define LEVO_CHAIN_LIBRARY
#include "ChainBench.cpp"
#include "core/TensorUtils.hpp"
#include <set>

int main(int argc,char** argv){
  if(argc!=2)return 2;
  Json result={{"success",false},{"packedActivation4KernelClaimed",false},{"purpose","Debug execution tensor quantization audit, not a speed test"}};
  try{
    const std::string root=argv[1],folder=root+"/smooth-a4/";
    auto config=readJson(folder+"config.json");
    config["base_dir"]=folder;config["llm_config"]="llm_config.json";config["embedding_file"]="embeddings_bf16.bin";
    config["thread_num"]=4;config["attention_mode"]=14;config["enable_debug"]=true;
    config["precision"]="low";config["memory"]="low";
    const std::string path=folder+"audit-config.json";{std::ofstream out(path);out<<config.dump();}
    std::unique_ptr<Llm,decltype(&Llm::destroy)> llm(Llm::createLLM(path),&Llm::destroy);
    require(llm&&llm->load(),"Audit LLM failed to load");
    Json samples=Json::array();int calls=0,integerTensorInputs=0,range4=0;std::set<std::string> names;
    llm->setDebugCallback([&](const std::vector<MNN::Tensor*>& tensors,const MNN::OperatorInfo* op){
      ++calls;
      for(auto* tensor:tensors){
        auto* desc=MNN::TensorUtils::getDescribe(tensor);auto quant=desc->quantAttr;
        const auto type=tensor->getType();
        if(type.code==halide_type_int&&type.bits==8)++integerTensorInputs;
        if(quant&&quant->min>=-8&&quant->max<=7){
          ++range4;
          if(samples.size()<12&&names.insert(op->name()).second)samples.push_back({{"op",op->name()},{"opType",op->type()},
            {"logicalRangeMin",quant->min},{"logicalRangeMax",quant->max},{"scale",quant->scale},
            {"tensorTypeCode",type.code},{"tensorTypeBits",type.bits}});
        }
      }
      return true;
    },[](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo*){return true;});
    auto prefix=readJson(root+"/fixture/fixture.json")["prefix"].get<std::vector<int>>();prefix.resize(64);
    llm->reset();llm->generate_init();auto output=llm->forward(prefix,true);
    require(output.get()&&output->getInfo(),"Audit forward failed");
    const float* data=output->readMap<float>();require(data,"Audit read failed");
    for(int i=0;i<output->getInfo()->size;++i)require(std::isfinite(data[i]),"Audit output non-finite");
    result["callbackCalls"]=calls;result["int8TensorInputObservations"]=integerTensorInputs;
    result["fourBitRangeInputObservations"]=range4;result["samples"]=samples;
    result["runtimeFourBitRangeObserved"]=range4>0;result["success"]=true;
  }catch(const std::exception& e){result["error"]=e.what();}
  std::cout<<"LEVO_A4_AUDIT "<<result.dump()<<std::endl;return result["success"].get<bool>()?0:1;
}
