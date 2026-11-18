#pragma once
// Isolated research operator. Registration requires the pinned-MNN lowp link
// wrapper in runtime/SnakeFusedLowp.cpp; enabled only by explicit graph rewrite.
#include "SnakeRuntime.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include <atomic>
#ifdef LEVO_VAE_ACTIVATION_RESEARCH
#include "VaeActivationResearch.hpp"
#endif

namespace LevoSnakeFused {
inline std::atomic<uint64_t> calls{0}, elements{0};
inline bool identifies(const MNN::Op* op) {
    auto p = op->main_as_Extra();
    return op->type() == MNN::OpType_Extra && p && p->engine() && p->type()
        && p->engine()->str() == "LevoResearch" && p->type()->str() == "SnakeFp16V1";
}
class Kernel final : public MNN::Execution {
    std::vector<float16_t> alpha_, beta_;
    size_t area_ = 0;
    int channels_ = 0;
    bool packed_ = false;
public:
    Kernel(MNN::Backend* backend, const MNN::Op* op) : Execution(backend) {
        auto info = op->main_as_Extra()->info();
        if (!info || info->size() == 0 || info->size() % 8 != 0 || info->size() > 2048 * 8)
            throw std::runtime_error("Invalid fused Snake coefficient storage");
        const size_t channels = info->size() / 8;
        alpha_.resize(channels); beta_.resize(channels);
        for (size_t c = 0; c < channels; ++c) {
            float a, b; std::memcpy(&a, info->data() + c * 4, 4); std::memcpy(&b, info->data() + (channels + c) * 4, 4);
            if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("Nonfinite Snake coefficient");
            alpha_[c] = float16_t(a); beta_[c] = float16_t(b);
        }
    }
    MNN::ErrorCode onResize(const std::vector<MNN::Tensor*>& inputs, const std::vector<MNN::Tensor*>& outputs) override {
        auto cpu = static_cast<MNN::CPUBackend*>(backend());
        if (inputs.size() != 1 || outputs.size() != 1 || cpu->functions()->bytes != 2) return MNN::NOT_SUPPORT;
        auto x = inputs[0], y = outputs[0];
        if (x->dimensions() < 3 || x->length(0) != 1 || x->shape() != y->shape() || x->length(1) != alpha_.size()) return MNN::INPUT_DATA_ERROR;
        channels_ = x->length(1); area_ = x->elementSize() / channels_;
        const auto format = MNN::TensorUtils::getDescribe(x)->dimensionFormat;
        if (format != MNN::TensorUtils::getDescribe(y)->dimensionFormat) return MNN::NOT_SUPPORT;
        packed_ = format == MNN::MNN_DATA_FORMAT_NC4HW4;
        if (packed_ && (cpu->functions()->pack != 8 || channels_ % 8 != 0)) return MNN::NOT_SUPPORT;
        if (!packed_ && format != MNN::MNN_DATA_FORMAT_NCHW) return MNN::NOT_SUPPORT;
        return MNN::NO_ERROR;
    }
    MNN::ErrorCode onExecute(const std::vector<MNN::Tensor*>& inputs, const std::vector<MNN::Tensor*>& outputs) override {
        auto cpu = static_cast<MNN::CPUBackend*>(backend());
        const auto x = inputs[0]->host<float16_t>(); auto y = outputs[0]->host<float16_t>();
        if (!x || !y) return MNN::INPUT_DATA_ERROR;
#ifdef LEVO_VAE_ACTIVATION_RESEARCH
        if(LevoVaeActivationResearch::selected(packed_,channels_,area_)){
            LevoVaeActivationResearch::execute(cpu,x,y,alpha_.data(),beta_.data(),channels_,area_);
            ++calls;elements+=size_t(channels_)*area_;return MNN::NO_ERROR;
        }
#endif
        // Tile along time as well as channels: final128-channel layers otherwise
        // have only16 packed rows and distribute poorly across6 workers.
        constexpr size_t tile = 2048;
        const size_t slices = (area_ + tile - 1) / tile, rows = packed_ ? channels_ / 8 : channels_;
        MNN::ThreadPool::TASK task; task.second = cpu->threadNumber();
        task.first = [&](int worker) {
            for (size_t job = worker; job < rows * slices; job += task.second) {
                const size_t row = job / slices, start = (job % slices) * tile, end = std::min(area_, start + tile);
                if (packed_) {
                    const auto a = vld1q_f16(alpha_.data() + row * 8), b = vld1q_f16(beta_.data() + row * 8);
                    const size_t base = row * area_ * 8;
                    for (size_t t = start; t < end; ++t)
                        vst1q_f16(y + base + t * 8, LevoVaeSnake::snake8(vld1q_f16(x + base + t * 8), a, b));
                } else {
                    const auto a = vdupq_n_f16(alpha_[row]), b = vdupq_n_f16(beta_[row]);
                    const size_t base = row * area_; size_t t = start;
                    for (; t + 8 <= end; t += 8) vst1q_f16(y + base + t, LevoVaeSnake::snake8(vld1q_f16(x + base + t), a, b));
                    for (; t < end; ++t) {
                        const auto z = float16_t(x[base + t] * alpha_[row]), sine = float16_t(std::sin(float(z)));
                        const auto square = float16_t(sine * sine), scaled = float16_t(square * beta_[row]);
                        y[base + t] = float16_t(x[base + t] + scaled);
                    }
                }
            }
        };
        cpu->enqueue(task); ++calls; elements += size_t(channels_) * area_; return MNN::NO_ERROR;
    }
};
class Creator final : public MNN::CPUBackend::Creator {
    MNN::Execution* onCreate(const std::vector<MNN::Tensor*>&, const std::vector<MNN::Tensor*>&, const MNN::Op* op, MNN::Backend* backend) const override {
        if (!identifies(op)) return nullptr;
        try { return new Kernel(backend, op); } catch (...) { return nullptr; }
    }
};
inline bool registerKernel() { static Creator creator; return MNN::CPUBackend::addCreator(MNN::OpType_Extra, &creator); }
}
