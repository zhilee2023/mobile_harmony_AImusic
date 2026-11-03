#pragma once
// Research-only API helper. Not enabled in either production App pipeline.
// Caller enables config["hidden_states"]=true before Llm::load(), forwards once,
// then copies this result before another forward/reset/destruction reuses it.
#include "llm/llm.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace LevoHiddenResearch {
inline std::vector<float> copySingleStep(MNN::Transformer::Llm& model, int expectedWidth) {
    if (expectedWidth <= 0) throw std::runtime_error("Hidden width must come from the training contract");
    const int index = model.getOutputIndex("hidden_states");
    const auto outputs = model.getOutputs();
    if (index < 0 || size_t(index) >= outputs.size() || outputs[index].get() == nullptr)
        throw std::runtime_error("Missing loaded hidden_states output; check graph and runtime configuration");
    const auto hidden = outputs[index];
    const auto info = hidden->getInfo();
    if (info == nullptr || info->dim.size() != 3 || info->dim[0] != 1 || info->dim[1] != 1 ||
        info->dim[2] != expectedWidth || info->size != expectedWidth)
        throw std::runtime_error("Expected canonical [1,1,H]; all-position/padded/block-prefill outputs need an explicit selector");
    if (info->type.code != halide_type_float || info->type.bits != 32)
        throw std::runtime_error("Research readMap<float> requires a float32 graph output");
    const float* data = hidden->readMap<float>();
    if (data == nullptr) throw std::runtime_error("Cannot synchronize hidden output");
    std::vector<float> owned(data, data+expectedWidth);
    for (float value : owned)
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite hidden condition");
    return owned;
}
}  // namespace LevoHiddenResearch
