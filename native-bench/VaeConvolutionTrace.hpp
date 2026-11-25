#pragma once

// Temporary instrumentation for the pinned MNN quantized convolution executor.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace LevoConvTrace {
using Clock = std::chrono::steady_clock;
enum Stage { Quant, Pack, GemmEpilogue, InputSum, OutputQuant, WeightReorder, Count };
struct alignas(64) Worker { std::array<uint64_t, Count> ns{}, calls{}; };

struct Trace {
    Clock::time_point begin = Clock::now();
    std::vector<Worker> slots;
    std::array<int, 15> shape;

    Trace(int threads, int e, int ic, int oc, int bits, int quant, int option, int split, int beforePack,
          int ep, int lp, int hp, int inputBytes, int outputBytes, int kernel, int stride)
        : slots(threads + 1), shape{e, ic, oc, bits, quant, option, split, beforePack,
                                   ep, lp, hp, inputBytes, outputBytes, kernel, stride} {}

    template <class F> void measure(int tid, Stage stage, F&& operation) {
        auto tick = Clock::now();
        operation();
        auto& slot = slots[tid];
        slot.ns[stage] += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tick).count();
        ++slot.calls[stage];
    }

    ~Trace() {
        const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
        std::printf("LEVO_VAE_DENSE_CONV");
        for (auto value : shape) std::printf(" %d", value);
        std::printf(" %llu", static_cast<unsigned long long>(wall));
        for (int stage = 0; stage < Count; ++stage) {
            uint64_t calls = slots.back().calls[stage], maximum = 0;
            for (size_t i = 0; i + 1 < slots.size(); ++i) {
                calls += slots[i].calls[stage];
                maximum = std::max(maximum, slots[i].ns[stage]);
            }
            std::printf(" %llu %llu %llu", static_cast<unsigned long long>(calls),
                        static_cast<unsigned long long>(slots.back().ns[stage]),
                        static_cast<unsigned long long>(maximum));
        }
        std::printf("\n");
    }
};
}
