#pragma once
// Opt-in CLI instrumentation. Never enabled in the product/experiment App.
#include <MNN/Interpreter.hpp>
#include <json.hpp>
#include <chrono>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace LevoStreamProfile {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
struct Node {
    std::string phase, name, type;
    uint64_t calls = 0;
    double seconds = 0, maximum = 0;
    std::vector<std::vector<int>> outputShapes;
};
struct Totals { uint64_t calls = 0; double seconds = 0, maximum = 0; };
struct Capture {
    bool operators = false;
    std::string phase, error;
    std::map<std::string, Node> nodes;
    std::map<std::string, Totals> spans;
    struct Pending { Node* node; Clock::time_point start; };
    std::vector<Pending> pending;
    MNN::TensorCallBackWithInfo before() {
        return [this](const std::vector<MNN::Tensor*>&, const MNN::OperatorInfo* info) {
            try {
                auto& node = nodes[phase + "|" + info->name()];
                if (node.name.empty()) { node.phase = phase; node.name = info->name(); node.type = info->type(); }
                pending.push_back({&node, Clock::now()});
            } catch (...) { error = "Profiler could not record before callback"; }
            return true; // Never skip model computation when diagnostics fail.
        };
    }
    MNN::TensorCallBackWithInfo after() {
        return [this](const std::vector<MNN::Tensor*>& tensors, const MNN::OperatorInfo*) {
            const auto end = Clock::now();
            try {
                if (pending.empty()) { error = "Unmatched operator callback"; return true; }
                auto timed = pending.back(); pending.pop_back();
                const double elapsed = std::chrono::duration<double>(end - timed.start).count();
                auto& node = *timed.node;
                if (!node.calls) for (auto t : tensors) node.outputShapes.push_back(t->shape());
                ++node.calls; node.seconds += elapsed; node.maximum = std::max(node.maximum, elapsed);
            } catch (...) { error = "Profiler could not record after callback"; }
            return true;
        };
    }
    Json report() const {
        Json result = {{"operatorCallbacksEnabled", operators}, {"error", error},
                       {"operators", Json::array()}, {"spans", Json::object()},
                       {"timingScope", "CPU callback wall time; inclusive nodes may overlap. Spans are nested, not additive; no kernel hardware counters."}};
        for (const auto& pair : spans) result["spans"][pair.first] = {{"calls", pair.second.calls}, {"seconds", pair.second.seconds}, {"maximumSeconds", pair.second.maximum}};
        for (const auto& pair : nodes) { const auto& n = pair.second;
            result["operators"].push_back({{"phase", n.phase}, {"name", n.name}, {"type", n.type}, {"calls", n.calls},
                                            {"seconds", n.seconds}, {"maximumSeconds", n.maximum}, {"outputShapes", n.outputShapes}});
        }
        std::sort(result["operators"].begin(), result["operators"].end(), [](const Json& a, const Json& b) { return a["seconds"].get<double>() > b["seconds"].get<double>(); });
        result["unmatchedCallbacks"] = pending.size(); return result;
    }
};
inline Capture* active = nullptr;
class Span {
    Capture* capture_ = active;
    std::string name_, previous_;
    Clock::time_point start_;
public:
    explicit Span(const std::string& name) : name_(name), start_(Clock::now()) {
        if (capture_) { previous_ = capture_->phase; capture_->phase = name; }
    }
    ~Span() {
        if (capture_) {
            auto& value = capture_->spans[name_]; ++value.calls;
            const double elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
            value.seconds += elapsed; value.maximum = std::max(value.maximum, elapsed); capture_->phase = previous_;
        }
    }
};
}
#define LEVO_PROFILE_JOIN2(a,b) a##b
#define LEVO_PROFILE_JOIN(a,b) LEVO_PROFILE_JOIN2(a,b)
#define LEVO_STREAM_SPAN(name) LevoStreamProfile::Span LEVO_PROFILE_JOIN(levoProfileSpan_,__COUNTER__)(name)
