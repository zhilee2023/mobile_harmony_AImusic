#pragma once

// Included in an isolated copy of the pinned MNN CPUDeconvolution.cpp.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>

namespace LevoDeconvTrace {
using Clock = std::chrono::steady_clock;
using Tick = Clock::time_point;
enum Stage { Clear, Pack, Gemm, Scatter, Reduce, Bias, Count };
struct alignas(64) Worker { std::array<uint64_t, Count> ns{}; };
struct Trace {
    std::array<Worker, 64> workers{};
    int threadCount = 0, elementBytes = 0, packWidth = 0, tile = 0;
    int kernel = 0, stride = 0, dilation = 0;
    size_t outputBytes = 0, packBytesPerWorker = 0, gemmBytesPerWorker = 0;
};
inline std::atomic<Trace*> current{nullptr};

inline uint64_t elapsed(Tick tick) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tick).count();
}
inline void mark(Worker& worker, Stage stage, Tick& tick) {
    const auto end = Clock::now();
    worker.ns[stage] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tick).count();
    tick = end;
}

inline void emit(const Trace& trace, int ic, int oc, int inElements, int outElements,
                 uint64_t parallelNs, uint64_t reductionNs) {
    std::printf("LEVO_VAE_DECONV %d %d %d %d %d %d %d %d %d %d %d %zu %zu %zu %llu %llu",
                ic, oc, inElements, outElements, trace.threadCount, trace.elementBytes,
                trace.packWidth, trace.tile, trace.kernel, trace.stride, trace.dilation,
                trace.outputBytes, trace.packBytesPerWorker, trace.gemmBytesPerWorker,
                static_cast<unsigned long long>(parallelNs), static_cast<unsigned long long>(reductionNs));
    for (int worker = 0; worker < trace.threadCount; ++worker) {
        for (auto ns : trace.workers[worker].ns) std::printf(" %llu", static_cast<unsigned long long>(ns));
    }
    std::printf("\n");
}
}
