#include "VaeSnakeRuntime.hpp"
#include <chrono>
#include <iostream>
#include <unistd.h>
using Json=nlohmann::json;
static std::vector<float> read(const std::string& path,size_t count) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);LevoVaeSnake::check(bool(in)&&in.tellg()==std::streamoff(count*4),"Fixture size mismatch");
    std::vector<float> result(count);in.seekg(0);in.read(reinterpret_cast<char*>(result.data()),count*4);return result;
}
int main(int argc,char** argv) {
    Json result={{"success",false},{"scope","Shared App Snake runtime, no preliminary warm capture; VAE only"}};
    try {
        LevoVaeSnake::check(argc==2,"Expected fixture directory");std::string dir=argv[1];
        std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile((dir+"/vae-256-w8.mnn").c_str()));LevoVaeSnake::check(bool(net),"Missing VAE");
        net->setExternalFile((dir+"/vae-256-w8.mnn.weight").c_str());net->setSessionMode(MNN::Interpreter::Session_Debug);net->setSessionMode(MNN::Interpreter::Session_Resize_Defer);net->setSessionHint(MNN::Interpreter::CPU_ENABLE_KLEIDIAI,0);
        MNN::BackendConfig backend;backend.precision=MNN::BackendConfig::Precision_Low;backend.memory=MNN::BackendConfig::Memory_Low;backend.power=MNN::BackendConfig::Power_High;
        MNN::ScheduleConfig config;config.type=MNN_FORWARD_CPU;config.numThread=6;config.backendConfig=&backend;
        auto* session=net->createSession(config);LevoVaeSnake::check(session,"Session unavailable");net->resizeSession(session);net->releaseModel();
        LevoVaeSnake::Runtime fusion(dir+"/vae-256-w8.mnn",dir+"/snake-coefficients.json");double total=0;
        for (int index=0;index<3;++index) {
            const auto values=read(dir+"/latent-"+std::to_string(index)+".f32",64*256);auto* input=net->getSessionInput(session,"latent");MNN::Tensor host(input,MNN::Tensor::CAFFE,true);std::memcpy(host.host<float>(),values.data(),values.size()*4);LevoVaeSnake::check(input->copyFromHostTensor(&host),"Input copy failed");
            const auto begin=std::chrono::steady_clock::now();fusion.run(net.get(),session);
            auto* output=net->getSessionOutput(session,"pcm");MNN::Tensor pcm(output,MNN::Tensor::CAFFE,true);LevoVaeSnake::check(output->copyToHostTensor(&pcm),"Output copy failed");const double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();total+=spent;
            const auto expected=read(dir+"/runtime-w8-"+std::to_string(index)+".f32",pcm.elementSize());double error=0,energy=0;
            for (size_t i=0;i<expected.size();++i) {const auto value=pcm.host<float>()[i];LevoVaeSnake::check(std::isfinite(value),"Nonfinite PCM");const double d=value-expected[i];error+=d*d;energy+=double(expected[i])*expected[i];}
            const double l2=std::sqrt(error/std::max(energy,1e-30));LevoVaeSnake::check(l2<.001,"Shared Snake runtime differs from W8 reference");
            result["rows"].push_back({{"window",index+1},{"seconds",spent},{"pcmL2",l2}});std::cout<<"LEVO_SNAKE_APP_WINDOW "<<result["rows"].back().dump()<<std::endl;
        }
        result["fusion"]=fusion.stats();result["computeSeconds"]=total;result["validPcmSeconds"]=25.6;result["vaeOnlyRtf"]=total/25.6;result["success"]=true;net->releaseSession(session);
    } catch (const std::exception& error) {result["error"]=error.what();}
    std::cout<<"LEVO_SNAKE_APP_REPORT "<<result.dump()<<std::endl;return result["success"].get<bool>()?0:1;
}
