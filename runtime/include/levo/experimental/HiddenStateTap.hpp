#pragma once
// Research-only API helper. Not enabled in either production App pipeline.
// Caller enables config["hidden_states"]=true before Llm::load(), forwards once,
// then copies this result before another forward/reset/destruction reuses it.
#include "llm/llm.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace LevoHiddenResearch {
enum class SingleStepLayout { Bth, Channels1x1 };
// Destination storage is owned by the caller and can be reused across tokens.
// MNN's getOutputs() still returns its own small vector; this is not a claim
// that the entire engine/API performs zero allocations.
inline void copySingleStepInto(MNN::Transformer::Llm& model, float* destination, int expectedWidth,
                               SingleStepLayout layout = SingleStepLayout::Bth) {
    if (expectedWidth <= 0) throw std::runtime_error("Hidden width must come from the training contract");
    if (destination == nullptr) throw std::runtime_error("Missing caller-owned hidden storage");
    const int index = model.getOutputIndex("hidden_states");
    const auto outputs = model.getOutputs();
    if (index < 0 || size_t(index) >= outputs.size() || outputs[index].get() == nullptr)
        throw std::runtime_error("Missing loaded hidden_states output; check graph and runtime configuration");
    const auto hidden = outputs[index];
    const auto info = hidden->getInfo();
    if (info == nullptr) throw std::runtime_error("Missing hidden metadata");
    const auto& dims = info->dim;
    const bool bth = dims.size()==3 && dims[0]==1 && dims[1]==1 && dims[2]==expectedWidth;
    const bool channels = dims.size()==4 && dims[0]==1 && dims[1]==expectedWidth && dims[2]==1 && dims[3]==1;
    // The converted LM can expose final RMSNorm in channel/1x1 form before
    // its Conv output head. At one spatial position NCHW/NC4 channel packing
    // has the same logical channel order. This opt-in never selects a frame
    // from multi-position or padded-prefill tensors.
    if (info->size != expectedWidth || !(layout == SingleStepLayout::Bth ? bth : channels))
        throw std::runtime_error("Hidden does not match the explicitly selected single-position layout");
    if (info->type.code != halide_type_float || info->type.bits != 32)
        throw std::runtime_error("Research readMap<float> requires a float32 graph output");
    const float* data = hidden->readMap<float>();
    if (data == nullptr) throw std::runtime_error("Cannot synchronize hidden output");
    for (int i = 0; i < expectedWidth; ++i) {
        if (!std::isfinite(data[i])) throw std::runtime_error("Nonfinite hidden condition");
        destination[i] = data[i];
    }
}

inline std::vector<float> copySingleStep(MNN::Transformer::Llm& model, int expectedWidth,
                                          SingleStepLayout layout = SingleStepLayout::Bth) {
    if (expectedWidth <= 0) throw std::runtime_error("Hidden width must come from the training contract");
    std::vector<float> owned(expectedWidth);
    copySingleStepInto(model, owned.data(), expectedWidth, layout);
    return owned;
}
}  // namespace LevoHiddenResearch
