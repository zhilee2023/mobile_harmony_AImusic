#pragma once
#include "../RuntimeSupport.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/TensorUtils.hpp"
namespace LlmOperatorLab {
struct MlpCapture {
    std::string root;
    Json phases=Json::object();
    explicit MlpCapture(std::string path):root(std::move(path)){if(!root.empty())std::filesystem::create_directories(root);}
    bool wants(const std::string& name)const{return !root.empty()&&name=="/blocks.0/mlp/Mul_output_0";}
    Json save(const std::string& phase,const std::string& role,MNN::Tensor* tensor){
        auto backend=MNN::TensorUtils::getDescribeOrigin(tensor)->getBackend();
        require(backend&&(int(backend->type())==0||int(backend->type())==13)&&MNN::CPUBackend::getBytes(backend,tensor)==4,"Capture requires actual FP32 CPU storage");
        require(tensor->elementSize()>0&&tensor->elementSize()<=513*9728,"Capture shape exceeds bound");
        MNN::Tensor host(tensor,MNN::Tensor::CAFFE,true);require(tensor->copyToHostTensor(&host),"Cannot copy actual MLP tensor");
        require(host.getType().code==halide_type_float&&host.getType().bits==32,"Capture is not FP32");
        const float* values=host.host<float>();for(int i=0;i<host.elementSize();++i)require(std::isfinite(values[i]),"Nonfinite capture");
        const std::string name=phase+"-"+role+".f32";std::ofstream out(root+"/"+name,std::ios::binary);out.write(reinterpret_cast<const char*>(values),host.elementSize()*4);require(out.good(),"Cannot save private MLP capture");
        return {{"file",name},{"shape",host.shape()},{"elements",host.elementSize()},{"bytes",host.elementSize()*4}};
    }
    void before(const std::string& phase,const std::string& name,const std::vector<MNN::Tensor*>& inputs){
        if(!wants(name)||phases.contains(phase))return;require(inputs.size()==2,"MUL_SILU requires two inputs");
        phases[phase]={{"input0",save(phase,"input0",inputs[0])},{"input1",save(phase,"input1",inputs[1])}};
    }
    void after(const std::string& phase,const std::string& name,const std::vector<MNN::Tensor*>& outputs){
        if(!wants(name)||!phases.contains(phase)||phases[phase].contains("output"))return;require(outputs.size()==1,"MUL_SILU output count changed");
        phases[phase]["output"]=save(phase,"output",outputs[0]);
    }
    Json report()const{return {{"phases",phases},{"expression","input0 * silu(input1)"},{"scope","Private actual first-block MLP activations, logical CAFFE host order; not weights, never publish tensor files"}};}
};
}
