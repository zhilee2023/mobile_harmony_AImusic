#pragma once
// Experimental callback fusion: retains the original graph workspace and one
// bounded producer snapshot. No process-global execution hook or hidden pool.
#include "VaeVectorSin.hpp"
#include "core/TensorUtils.hpp"
#include <MNN/Interpreter.hpp>
#include <json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace LevoVaeSnake {
using Json = nlohmann::json;
inline void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline bool ends(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size()-suffix.size(), suffix.size(), suffix)==0;
}
inline MNN::Tensor* storage(MNN::Tensor* tensor) {
    for (int depth=0; depth<4; ++depth) {
        auto* description=MNN::TensorUtils::getDescribe(tensor);
        if (description->memoryType!=MNN::Tensor::InsideDescribe::MEMORY_VIRTUAL) return tensor;
        check(description->regions.size()==1, "Snake requires one complete storage view");
        const auto& r=description->regions[0];
        const size_t count=size_t(r.size[0])*r.size[1]*r.size[2];
        check(r.origin && r.src.offset==0 && r.dst.offset==0 && MNN::TensorUtils::isCopyRegion(r), "Snake refuses sliced/transposed storage");
        check(count==tensor->elementSize() && count==r.origin->elementSize() && tensor->length(1)==r.origin->length(1), "Snake requires complete channel-preserving view");
        tensor=r.origin;
    }
    throw std::runtime_error("Snake storage view depth exceeds bound");
}
inline float16x8_t snake8(float16x8_t x, float16x8_t alpha, float16x8_t inverseBeta) {
#pragma clang fp contract(off)
    const auto z=vmulq_f16(x,alpha);
    float16x8_t sine;
    if (vmaxvq_f16(vabsq_f16(z))>128) {
        alignas(16) float16_t values[8]; vst1q_f16(values,z);
        for (auto& value: values) value=float16_t(std::sin(float(value)));
        sine=vld1q_f16(values);
    } else {
        sine=vcombine_f16(vcvt_f16_f32(LevoVaeSin::sine4(vcvt_f32_f16(vget_low_f16(z)))),
                          vcvt_f16_f32(LevoVaeSin::sine4(vcvt_f32_f16(vget_high_f16(z)))));
    }
    const auto square=vmulq_f16(sine,sine);
    const auto scaled=vmulq_f16(square,inverseBeta);
    return vaddq_f16(x,scaled);
}

class Runtime {
    struct Group {
        int originalInput=-1;
        MNN::Tensor* input=nullptr;
        std::string producer;
        std::vector<float> alpha,inverseBeta;
    };
    std::map<std::string,Group> groups;
    std::map<std::string,std::string> producers;
    std::vector<float16_t> snapshot;
    std::string pending;
    MNN::MNN_DATA_FORMAT inputFormat=MNN::MNN_DATA_FORMAT_NCHW;
    size_t snapshotPeak=0,calls=0,forwards=0;
    static constexpr size_t maxElements=size_t(128)*491520;

    void capture(Group& group, const std::string& name, MNN::Tensor* tensor) {
        check(pending.empty(), "Snake producer snapshots overlap");
        auto* source=storage(tensor);
        auto* backend=MNN::TensorUtils::getDescribeOrigin(source)->getBackend();
        check(backend && MNN::CPUBackend::getBytes(backend,source)==2, "Snake producer requires FP16");
        check(source->length(1)==group.alpha.size(), "Snake producer channel mismatch");
        const size_t count=source->elementSize();
        check(count>0 && count<=maxElements && source->host<float16_t>(), "Snake producer exceeds memory bound");
        inputFormat=MNN::TensorUtils::getDescribe(source)->dimensionFormat;
        snapshot.resize(count);
        std::memcpy(snapshot.data(),source->host<float16_t>(),count*2);
        pending=name;snapshotPeak=std::max(snapshotPeak,count*2);
    }
    void compute(Group& group, MNN::Tensor* tensor) {
        auto* output=storage(tensor);
        auto* backend=MNN::TensorUtils::getDescribeOrigin(output)->getBackend();
        check(backend && MNN::CPUBackend::getBytes(backend,output)==2, "Snake output requires FP16");
        auto* cpu=static_cast<MNN::CPUBackend*>(backend);
        const int channels=group.input->length(1),time=group.input->length(2);
        check(channels>0 && time>0 && group.input->elementSize()==size_t(channels)*time && output->elementSize()==size_t(channels)*time, "Snake requires one batch and unchanged shape");
        check(group.alpha.size()==channels && snapshot.size()==size_t(channels)*time, "Snake coefficient/snapshot shape mismatch");
        const auto outputFormat=MNN::TensorUtils::getDescribe(output)->dimensionFormat;
        const bool packed=inputFormat==MNN::MNN_DATA_FORMAT_NC4HW4 && outputFormat==MNN::MNN_DATA_FORMAT_NC4HW4;
        check(packed || (inputFormat==MNN::MNN_DATA_FORMAT_NCHW && outputFormat==MNN::MNN_DATA_FORMAT_NCHW), "Snake requires matching layouts");
        if (packed) check(cpu->functions()->bytes==2 && cpu->functions()->pack==8 && channels%8==0, "Snake requires matching C8 ABI");
        const auto* x=snapshot.data();auto* y=output->host<float16_t>();check(y,"Snake output unavailable");
        MNN::ThreadPool::TASK task;task.second=6;
        task.first=[&](int worker) {
#pragma clang fp contract(off)
            if (packed) {
                for (int c=worker*8;c<channels;c+=48) {
                    alignas(16) float16_t a[8],b[8];
                    for (int k=0;k<8;++k) {a[k]=float16_t(group.alpha[c+k]);b[k]=float16_t(group.inverseBeta[c+k]);}
                    const auto av=vld1q_f16(a),bv=vld1q_f16(b);const size_t base=size_t(c/8)*time*8;
                    for (int t=0;t<time;++t) vst1q_f16(y+base+size_t(t)*8,snake8(vld1q_f16(x+base+size_t(t)*8),av,bv));
                }
            } else {
                for (int c=worker;c<channels;c+=6) {
                    const auto av=vdupq_n_f16(float16_t(group.alpha[c])),bv=vdupq_n_f16(float16_t(group.inverseBeta[c]));
                    int t=0;
                    for (;t+8<=time;t+=8) vst1q_f16(y+size_t(c)*time+t,snake8(vld1q_f16(x+size_t(c)*time+t),av,bv));
                    for (;t<time;++t) {
                        const auto z=float16_t(x[size_t(c)*time+t]*float16_t(group.alpha[c]));
                        const auto sine=float16_t(std::sin(float(z))),square=float16_t(sine*sine);
                        const auto scaled=float16_t(square*float16_t(group.inverseBeta[c]));
                        y[size_t(c)*time+t]=float16_t(x[size_t(c)*time+t]+scaled);
                    }
                }
            }
        };
        cpu->enqueue(task);
    }
public:
    Runtime(const std::string& modelPath,const std::string& coefficientPath) {
        std::ifstream file(modelPath,std::ios::binary|std::ios::ate);
        check(bool(file) && file.tellg()>0 && file.tellg()<4*1024*1024,"Snake graph metadata unavailable or unbounded");
        std::vector<char> data(size_t(file.tellg()));file.seekg(0);file.read(data.data(),data.size());check(file.good(),"Snake graph metadata read failed");
        const auto* model=MNN::GetNet(data.data());
        Json coefficients;std::ifstream input(coefficientPath);check(bool(input),"Missing snake-coefficients.json");input>>coefficients;
        check(coefficients.at("groups").size()==36,"Expected 36 Snake coefficient groups");
        for (const auto* op:*model->oplists()) {
            const auto name=op->name()->str();const std::string suffix="/Add_1_output_0";
            if (!ends(name,suffix)) continue;
            const auto prefix=name.substr(0,name.size()-suffix.size());auto& group=groups[prefix];
            check(op->inputIndexes() && op->inputIndexes()->size()==2,"Invalid Snake residual Add");
            for (int i=0;i<2;++i) if (ends(model->tensorName()->Get(op->inputIndexes()->Get(i))->str(),"/Mul_2_output_0")) group.originalInput=1-i;
            check(group.originalInput>=0,"Snake residual input not identified");
            int source=op->inputIndexes()->Get(group.originalInput);
            for (int depth=0;depth<8;++depth) {
                const MNN::Op* matched=nullptr;
                for (const auto* candidate:*model->oplists()) if (candidate->outputIndexes())
                    for (auto id:*candidate->outputIndexes()) if (id==source) matched=candidate;
                check(matched,"Snake producer missing");
                if (matched->type()==MNN::OpType_Squeeze || matched->type()==MNN::OpType_Unsqueeze || matched->type()==MNN::OpType_Reshape) {
                    check(matched->inputIndexes() && matched->inputIndexes()->size()>0,"Invalid Snake producer view");source=matched->inputIndexes()->Get(0);continue;
                }
                group.producer=matched->name()->str();break;
            }
            check(!group.producer.empty() && !producers.count(group.producer),"Snake producer ambiguous");producers[group.producer]=prefix;
            const auto& values=coefficients.at("groups").at(prefix);
            group.alpha=values.at("alpha").get<std::vector<float>>();group.inverseBeta=values.at("beta").get<std::vector<float>>();
            check(!group.alpha.empty() && group.alpha.size()<=2048 && group.alpha.size()==group.inverseBeta.size(),"Snake coefficients exceed bound");
            for (float value:group.alpha) check(std::isfinite(value),"Nonfinite Snake alpha");
            for (float value:group.inverseBeta) check(std::isfinite(value),"Nonfinite Snake inverse beta");
        }
        check(groups.size()==36,"Expected full VAE Snake graph");
    }
    void run(MNN::Interpreter* net,MNN::Session* session) {
        std::string error;const size_t previous=calls;pending.clear();
        auto before=[&](const std::vector<MNN::Tensor*>& inputs,const MNN::OperatorInfo* info) {
            if (!error.empty()) return false;
            try {
                const std::string name=info->name();const auto slash=name.rfind('/');if (slash==std::string::npos) return true;
                auto found=groups.find(name.substr(0,slash));if (found==groups.end()) return true;
                if (ends(name,"/Add_1_output_0")) { found->second.input=inputs.at(found->second.originalInput);return false; }
                return !(ends(name,"/Mul_1_output_0") || ends(name,"/Mul_2_output_0") || ends(name,"/Sin_output_0") || ends(name,"/Pow_output_0"));
            } catch (const std::exception& e) {error=e.what();return false;}
        };
        auto after=[&](const std::vector<MNN::Tensor*>& outputs,const MNN::OperatorInfo* info) {
            if (!error.empty()) return false;
            try {
                const std::string name=info->name();const std::string suffix="/Add_1_output_0";
                if (ends(name,suffix)) {
                    const auto prefix=name.substr(0,name.size()-suffix.size());auto found=groups.find(prefix);
                    if (found!=groups.end()) {
                        check(pending==prefix && found->second.input,"Snake current producer snapshot missing");
                        compute(found->second,outputs.at(0));pending.clear();snapshot.clear();++calls;
                    }
                }
                auto producer=producers.find(name);
                if (producer!=producers.end()) capture(groups.at(producer->second),producer->second,outputs.at(0));
                return true;
            } catch (const std::exception& e) {error=e.what();return false;}
        };
        const auto code=net->runSessionWithCallBackInfo(session,before,after,true);
        if (!error.empty()) throw std::runtime_error(error);
        check(code==MNN::NO_ERROR && pending.empty() && calls-previous==36,"Incomplete Snake callback forward");++forwards;
    }
    Json stats() const {return {{"fusedCalls",calls},{"completedForwards",forwards},{"groups",groups.size()},
        {"producerSnapshotPeakBytes",snapshotPeak},{"originalGraphWorkspaceRetained",true},{"requiresWarmCapture",false}};}
};
} // namespace LevoVaeSnake
