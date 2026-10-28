#pragma once
// Optional deployment building block. Requires exported and PC-validated music
// graphs; this does not convert a text LLM to a music model.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

class LevoMnnGraph {
    std::unique_ptr<MNN::Interpreter> net;
    MNN::Session* session=nullptr;
public:
    explicit LevoMnnGraph(const std::string& path,int threads=4) {
        net.reset(MNN::Interpreter::createFromFile(path.c_str()));
        if(!net)throw std::runtime_error("Cannot read music MNN graph");
        MNN::BackendConfig backend;
        backend.precision=MNN::BackendConfig::Precision_High;
        backend.memory=MNN::BackendConfig::Memory_Low;
        MNN::ScheduleConfig schedule;
        schedule.type=MNN_FORWARD_CPU; schedule.numThread=threads; schedule.backendConfig=&backend;
        net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);
        session=net->createSession(schedule);
        if(!session)throw std::runtime_error("Cannot create music MNN session");
    }
    ~LevoMnnGraph(){if(session)net->releaseSession(session);}
    void shape(const char* name,const std::vector<int>& dims){
        auto tensor=net->getSessionInput(session,name);
        if(!tensor)throw std::runtime_error("Missing graph input");
        net->resizeTensor(tensor,dims);
    }
    void resize(){net->resizeSession(session);net->releaseModel();}
    void input(const char* name,const std::vector<float>& values){
        auto tensor=net->getSessionInput(session,name);
        if(!tensor||tensor->elementSize()!=values.size())throw std::runtime_error("Input shape mismatch");
        MNN::Tensor host(tensor,MNN::Tensor::CAFFE);
        std::copy(values.begin(),values.end(),host.host<float>());
        tensor->copyFromHostTensor(&host);
    }
    std::vector<float> run(const char* output){
        if(net->runSession(session)!=MNN::NO_ERROR)throw std::runtime_error("MNN run failed");
        auto tensor=net->getSessionOutput(session,output);
        if(!tensor)throw std::runtime_error("Missing graph output");
        MNN::Tensor host(tensor,MNN::Tensor::CAFFE); tensor->copyToHostTensor(&host);
        std::vector<float> values(host.host<float>(),host.host<float>()+host.elementSize());
        for(float x:values)if(!std::isfinite(x))throw std::runtime_error("Non-finite music output");
        return values;
    }
};
