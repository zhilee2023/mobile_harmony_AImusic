#include "VaeLayoutAlias.hpp"

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Pipeline.hpp"
#include <MNN_generated.h>
#include <set>
#include <stdexcept>

namespace LevoVaeAlias {
namespace {
std::set<std::string> requested, active;
std::string error;
uint64_t executions = 0;

bool isFullCopy(const MNN::CPUBackend* cpu, const MNN::Tensor* src, const MNN::Tensor* dst) {
    using MNN::TensorUtils;
    auto x = TensorUtils::getDescribe(src), y = TensorUtils::getDescribe(dst);
    if (x->usage != MNN::TensorUsage::NORMAL || y->usage != MNN::TensorUsage::NORMAL ||
        x->dimensionFormat != MNN::MNN_DATA_FORMAT_NC4HW4 || x->dimensionFormat != y->dimensionFormat ||
        x->applyQuant || y->applyQuant || src->getType() != dst->getType() ||
        src->getType().code != halide_type_float || cpu->functions()->bytes != 2 || cpu->functions()->pack != 8 ||
        src->dimensions() < 3 || dst->dimensions() < 3 || src->length(0) != 1 || dst->length(0) != 1 ||
        src->length(1) != dst->length(1) || src->length(1) % 8 || src->elementSize() != dst->elementSize() ||
        cpu->getTensorSize(src, true) != cpu->getTensorSize(dst, true) || y->regions.size() != 1) return false;
    const auto& region = y->regions.front();
    return region.origin == src && region.src.offset == 0 && region.dst.offset == 0 &&
           region.size[0] == 1 && region.size[1] == 1 && region.size[2] == dst->elementSize() &&
           region.src.stride[2] == 1 && region.dst.stride[2] == 1;
}

class AliasRaster final : public MNN::Execution {
public:
    explicit AliasRaster(MNN::Backend* backend) : Execution(backend) { mNeedAllocIO = false; }

    MNN::ErrorCode onResize(const std::vector<MNN::Tensor*>& inputs,
                            const std::vector<MNN::Tensor*>& outputs) override {
        using MNN::TensorUtils;
        if (inputs.size() != 1 || outputs.size() != 1 ||
            !isFullCopy(static_cast<MNN::CPUBackend*>(backend()), inputs.front(), outputs.front())) return MNN::NOT_SUPPORT;
        auto src = TensorUtils::getDescribeOrigin(inputs.front());
        auto dst = TensorUtils::getDescribeOrigin(outputs.front());
        if (!src->mem.get() || dst->mem.get() || src->getBackend() != backend() || dst->getBackend() != backend()) {
            error = "Unexpected allocation or backend for alias";
            return MNN::NOT_SUPPORT;
        }
        // An additional reference keeps the original allocation live through
        // the last consumer of the view. Shape metadata stays independent.
        TensorUtils::setLinearLayout(outputs.front());
        dst->mem = src->mem;
        dst->offset = src->offset;
        outputs.front()->buffer().host = inputs.front()->buffer().host;
        return MNN::NO_ERROR;
    }

    MNN::ErrorCode onExecute(const std::vector<MNN::Tensor*>& inputs,
                             const std::vector<MNN::Tensor*>& outputs) override {
        auto ptr = inputs.front()->buffer().host;
        if (!ptr) return MNN::INPUT_DATA_ERROR;
        // Deferred CPU allocation attaches the final address to the source
        // tensor at execute begin. Propagate it to the shape view here.
        outputs.front()->buffer().host = ptr;
        ++executions;
        return MNN::NO_ERROR;
    }
};
}

void configure(const std::vector<std::string>& names) {
    requested = {names.begin(), names.end()};
    if (names.size() != requested.size() || requested.size() > 72) throw std::runtime_error("Invalid alias plan");
    active.clear();
    error.clear();
    executions = 0;
}
std::vector<std::string> installed() { return {active.begin(), active.end()}; }
std::string setupError() { return error; }
uint64_t calls() { return executions; }
}

extern "C" MNN::ErrorCode __real__ZN3MNN8Pipeline11allocMemoryEbb(MNN::Pipeline*, bool, bool);
extern "C" MNN::ErrorCode __wrap__ZN3MNN8Pipeline11allocMemoryEbb(MNN::Pipeline* pipeline,
                                                                bool firstMalloc, bool forbidReplace) {
    using namespace LevoVaeAlias;
    if (!requested.empty() && firstMalloc) {
        auto& info = pipeline->getPipelineInfo();
        auto* backend = info.first.cache.first.get();
        if (!backend || backend->type() != MNN_FORWARD_CPU_EXTENSION) {
            error = "Requires the pinned CPU Low backend";
            return MNN::NOT_SUPPORT;
        }
        auto* cpu = static_cast<MNN::CPUBackend*>(backend);
        for (auto& op : info.second) {
            if (op.type == MNN::Schedule::CONSTANT) continue;
            for (auto& cmd : op.executeBuffer.command) {
                if (!cmd->info || !requested.count(cmd->info->name())) continue;
                if (cmd->op->type() != MNN::OpType_Raster || cmd->inputs.size() != 1 || cmd->outputs.size() != 1 ||
                    !isFullCopy(cpu, cmd->inputs.front(), cmd->outputs.front()) ||
                    !active.insert(cmd->info->name()).second) {
                    error = "Raster contract mismatch: " + cmd->info->name();
                    return MNN::NOT_SUPPORT;
                }
                cmd->execution = std::make_shared<AliasRaster>(cpu);
            }
        }
        if (active != requested) {
            error = "Expected Raster was absent after geometry planning";
            return MNN::NOT_SUPPORT;
        }
    }
    const auto status = __real__ZN3MNN8Pipeline11allocMemoryEbb(pipeline, firstMalloc, forbidReplace);
    if (!requested.empty() && status != MNN::NO_ERROR && error.empty()) error = "Aliased session allocation failed";
    return status;
}
