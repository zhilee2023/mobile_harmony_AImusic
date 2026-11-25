// Dedicated experimental binary. The pinned archive builds Plugin with no implementation.
// Use its fixed registration symbols to load only this exact graph plugin in the lab.
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "shape/SizeComputer.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>

namespace {
using MNN::CPUBackend;
constexpr int Half = 3072;
constexpr int Tile = 2048;
std::atomic<uint64_t> calls{0};

bool isGate(const MNN::Op* op) {
    const auto* plugin = op->main_as_Plugin();
    return plugin && plugin->type() && plugin->type()->str() == "LevoPairedW4MulSilu";
}

bool hasShape(const MNN::Tensor* tensor, int channels) {
    const auto type = tensor->getType();
    return tensor->dimensions() == 4 && tensor->length(0) > 0 &&
        tensor->length(1) == channels && tensor->length(2) == 1 && tensor->length(3) == 1 &&
        type.code == halide_type_float && type.bits == 32 &&
        MNN::TensorUtils::getDescribe(tensor)->dimensionFormat == MNN::MNN_DATA_FORMAT_NC4HW4;
}

class GateShape : public MNN::SizeComputer {
public:
    bool onComputeSize(const MNN::Op* op, const std::vector<MNN::Tensor*>& inputs,
                       const std::vector<MNN::Tensor*>& outputs) const override {
        if (!isGate(op) || inputs.size() != 1 || outputs.size() != 1 || !hasShape(inputs[0], 2 * Half)) {
            return false;
        }
        auto output = outputs.front();
        MNN::TensorUtils::copyShape(inputs.front(), output, true, false);
        output->buffer().type = inputs.front()->getType();
        output->setLength(1, Half);
        MNN::TensorUtils::setLinearLayout(output);
        return true;
    }
};

class GateExecution : public MNN::Execution {
public:
    explicit GateExecution(MNN::CPUBackend* backend) : MNN::Execution(backend), cpu(backend) {
        const auto* core = cpu->functions();
        silu = core->MNNSelectUnaryFunctionForFloat(MNN::UnaryOpOperation_SILU, cpu->precisionMode());
        multiply = core->MNNSelectBinaryFunctionForFloat(MNN::BinaryOpOperation_MUL);
    }

    MNN::ErrorCode onResize(const std::vector<MNN::Tensor*>& inputs,
                            const std::vector<MNN::Tensor*>& outputs) override {
        const auto* core = cpu->functions();
        if (!silu || !multiply || core->bytes != 4 || core->pack != 4 ||
            !hasShape(inputs[0], 2 * Half) || !hasShape(outputs[0], Half) ||
            outputs[0]->length(0) > std::numeric_limits<int>::max() / (2 * Half) ||
            inputs[0]->length(0) != outputs[0]->length(0)) {
            return MNN::NOT_SUPPORT;
        }
        size = Half * outputs[0]->length(0);
        if (cpu->getTensorSize(outputs[0]) != size_t(size) ||
            cpu->getTensorSize(inputs[0]) != size_t(2) * size) return MNN::NOT_SUPPORT;
        return MNN::NO_ERROR;
    }

    MNN::ErrorCode onExecute(const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) override {
        const auto* up = inputs[0]->host<float>();
        auto* result = outputs[0]->host<float>();
        if (!up || !result) return MNN::INVALID_VALUE;
        // CPU NC4HW4 stores channel packs outside the T axis. Because 3072
        // is pack-aligned, the second projection starts at Half * T.
        const auto* gate = up + size;
        const uintptr_t in = reinterpret_cast<uintptr_t>(up);
        const uintptr_t out = reinterpret_cast<uintptr_t>(result);
        if (out < in + 2 * size * sizeof(float) && in < out + size * sizeof(float)) {
            return MNN::INVALID_VALUE;
        }
        const auto schedule = cpu->multiThreadDivide(size);
        MNN_CONCURRENCY_BEGIN(tid, schedule.second) {
            const auto begin = int(tid) * schedule.first;
            const auto end = tid == schedule.second - 1 ? size : begin + schedule.first;
            for (int offset = begin; offset < end; offset += Tile) {
                const int count = std::min(Tile, end - offset);
                silu(result + offset, gate + offset, count);
                multiply(result + offset, result + offset, up + offset, count, -1);
            }
        }
        MNN_CONCURRENCY_END();
        ++calls;
        return MNN::NO_ERROR;
    }
private:
    MNN::CPUBackend* cpu;
    MNNUnaryExecute silu = nullptr;
    MNNBinaryExecute multiply = nullptr;
    int size = 0;
};

class GateCreator : public MNN::CPUBackend::Creator {
public:
    MNN::Execution* onCreate(const std::vector<MNN::Tensor*>&, const std::vector<MNN::Tensor*>&,
                             const MNN::Op* op, MNN::Backend* backend) const override {
        return isGate(op) ? new GateExecution(static_cast<MNN::CPUBackend*>(backend)) : nullptr;
    }
};
}

namespace MNN {
void ___PluginSizeComputer__OpType_Plugin__() {
    SizeComputerSuite::get()->insert(new GateShape, OpType_Plugin);
}

void ___CPUPluginCreator__OpType_Plugin__() {
    static GateCreator creator;
    CPUBackend::addCreator(OpType_Plugin, &creator);
}
}

extern "C" uint64_t LlmOperatorCandidateCalls() {
    return calls.load();
}
