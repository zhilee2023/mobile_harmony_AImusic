#pragma once

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

namespace LevoDeconvEpilogue {

// The ARM82 baseline adds partials to the destination in worker order, then
// adds bias and clamps. Keep each FP16 addition in that order in a single pass.
inline void run(uint8_t* output, const uint8_t* partials, std::size_t partialStrideBytes,
                const uint8_t* bias, std::size_t positions, const float* parameters) {
    auto* destination = reinterpret_cast<float16_t*>(output);
    const auto* first = reinterpret_cast<const float16_t*>(partials);
    const auto* second = reinterpret_cast<const float16_t*>(partials + partialStrideBytes);
    const auto* third = reinterpret_cast<const float16_t*>(partials + 2 * partialStrideBytes);
    const auto biasValue = vld1q_f16(reinterpret_cast<const float16_t*>(bias));
    const auto minimum = vdupq_n_f16(parameters[2]);
    const auto maximum = vdupq_n_f16(parameters[3]);

    for (std::size_t offset = 0; offset < positions * 8; offset += 8) {
        auto sum = vaddq_f16(vld1q_f16(first + offset), vld1q_f16(destination + offset));
        sum = vaddq_f16(vld1q_f16(second + offset), sum);
        sum = vaddq_f16(vld1q_f16(third + offset), sum);
        sum = vaddq_f16(sum, biasValue);
        vst1q_f16(destination + offset, vmaxq_f16(vminq_f16(sum, maximum), minimum));
    }
}

}
