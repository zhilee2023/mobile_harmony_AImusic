#pragma once
// Optional FP32 conditioner primitive. Supply matching TRAINED weights in real
// use. Only frame-independent affine layers may be batched this way; this is
// not a replacement for a temporal/attention/recurrent conditioner.
#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace LevoHiddenResearch {
class FramewiseAffine {
    MNN::Express::VARP input_, output_;
    size_t inputCount_, outputCount_;
public:
    FramewiseAffine(int frames, int hiddenWidth, int conditionWidth,
                    const std::vector<float>& weight, const std::vector<float>& bias) {
        using namespace MNN::Express;
        if(frames<=0 || hiddenWidth<=0 || conditionWidth<=0 ||
           weight.size()!=size_t(hiddenWidth)*conditionWidth || bias.size()!=size_t(conditionWidth))
            throw std::runtime_error("Affine trained weight/shape contract mismatch");
        for(float x:weight)if(!std::isfinite(x))throw std::runtime_error("Nonfinite affine weight");
        for(float x:bias)if(!std::isfinite(x))throw std::runtime_error("Nonfinite affine bias");
        inputCount_=size_t(frames)*hiddenWidth;outputCount_=size_t(frames)*conditionWidth;
        input_=_Input({frames,hiddenWidth},NCHW,halide_type_of<float>());
        auto w=_Const(weight.data(),{conditionWidth,hiddenWidth},NCHW,halide_type_of<float>());
        auto b=_Const(bias.data(),{conditionWidth},NCHW,halide_type_of<float>());
        output_=_Add(_MatMul(input_,w,false,true),b);
    }
    // Own destination storage; graph and packed constant weights are reused.
    // Configure the executor in the caller, before creating this primitive.
    void runInto(const float* input,size_t inputElements,float* destination,size_t outputElements) {
        if(!input || !destination || inputElements!=inputCount_ || outputElements!=outputCount_)
            throw std::runtime_error("Affine input/output size mismatch");
        auto target=input_->writeMap<float>();
        if(!target)throw std::runtime_error("Cannot map affine input");
        std::copy_n(input,inputCount_,target);
        const float* values=output_->readMap<float>();
        if(!values || !output_->getInfo() || size_t(output_->getInfo()->size)!=outputCount_)
            throw std::runtime_error("Affine output contract mismatch");
        for(size_t i=0;i<outputCount_;++i){
            if(!std::isfinite(values[i]))throw std::runtime_error("Nonfinite affine condition");
            destination[i]=values[i];
        }
    }
};
}
