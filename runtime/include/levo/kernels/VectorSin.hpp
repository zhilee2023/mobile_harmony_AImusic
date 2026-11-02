#pragma once
// Experimental standalone benchmark hook. No global MNN source modification.
// Use FP32 range reduction/polynomials for eight FP16 samples at once. Large
// arguments retain scalar libm behavior; the original selector is restored.
#include "backend/arm82/Arm82Functions.hpp"
#include "backend/arm82/Arm82Unary.hpp"
#include <MNN_generated.h>
#include <arm_neon.h>
#include <atomic>
#include <cmath>
#include <stdexcept>
namespace LevoVaeSin {
inline std::atomic<unsigned long> calls{0};
inline decltype(MNN::CoreFunctions::MNNSelectUnaryFunctionForFloat) original=nullptr;
inline float32x4_t sine4(float32x4_t x){
  auto q=vcvtnq_s32_f32(vmulq_n_f32(x,0.63661977236758134f));
  auto qf=vcvtq_f32_s32(q);
  auto r=vfmsq_n_f32(x,qf,1.57079625129699707f);
  r=vfmsq_n_f32(r,qf,7.54978941586159635e-8f);
  auto r2=vmulq_f32(r,r);
  auto s=vdupq_n_f32(1.0f/362880.0f);
  s=vfmaq_f32(vdupq_n_f32(-1.0f/5040.0f),s,r2);
  s=vfmaq_f32(vdupq_n_f32(1.0f/120.0f),s,r2);
  s=vfmaq_f32(vdupq_n_f32(-1.0f/6.0f),s,r2);
  s=vfmaq_f32(r,vmulq_f32(r,r2),s);
  auto c=vdupq_n_f32(-1.0f/3628800.0f);
  c=vfmaq_f32(vdupq_n_f32(1.0f/40320.0f),c,r2);
  c=vfmaq_f32(vdupq_n_f32(-1.0f/720.0f),c,r2);
  c=vfmaq_f32(vdupq_n_f32(1.0f/24.0f),c,r2);
  c=vfmaq_f32(vdupq_n_f32(-0.5f),c,r2);
  c=vfmaq_f32(vdupq_n_f32(1.0f),c,r2);
  auto odd=vtstq_s32(q,vdupq_n_s32(1));
  auto y=vbslq_f32(odd,c,s);
  auto sign=vshlq_n_u32(vandq_u32(vreinterpretq_u32_s32(q),vdupq_n_u32(2)),30);
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(y),sign));
}
inline void execute(void* destination,const void* source,int size){
  ++calls;auto* out=static_cast<float16_t*>(destination);auto* in=static_cast<const float16_t*>(source);int i=0;
  for(;i+8<=size;i+=8){auto x=vld1q_f16(in+i);
    if(vmaxvq_f16(vabsq_f16(x))>128.0f){for(int k=0;k<8;++k)out[i+k]=float16_t(std::sin(float(in[i+k])));continue;}
    auto low=sine4(vcvt_f32_f16(vget_low_f16(x))),high=sine4(vcvt_f32_f16(vget_high_f16(x)));
    vst1q_f16(out+i,vcombine_f16(vcvt_f16_f32(low),vcvt_f16_f32(high)));
  }
  for(;i<size;++i)out[i]=float16_t(std::sin(float(in[i])));
}
inline MNNUnaryExecute select(int type,int precision){return type==MNN::UnaryOpOperation_SIN?execute:original(type,precision);}
struct SelfTest {int finite=0,different=0;float maxAbsoluteError=0;};
inline SelfTest selfTest(){
  alignas(16) uint16_t bits[8];alignas(16) float16_t output[8];SelfTest result;
  for(unsigned base=0;base<65536;base+=8){for(unsigned k=0;k<8;++k)bits[k]=uint16_t(base+k);execute(output,bits,8);
    auto* input=reinterpret_cast<float16_t*>(bits);
    for(int k=0;k<8;++k){float x=input[k];if(!std::isfinite(x))continue;float expected=float(float16_t(std::sin(x)));float error=std::abs(float(output[k])-expected);if(!std::isfinite(float(output[k])))throw std::runtime_error("Vector SIN produced nonfinite output");++result.finite;if(error>0)++result.different;result.maxAbsoluteError=std::max(result.maxAbsoluteError,error);}
  }calls=0;if(result.maxAbsoluteError>.001f)throw std::runtime_error("Vector SIN exhaustive FP16 error bound failed");return result;
}
struct Scope {
  MNN::CoreFunctions* core=nullptr;
  explicit Scope(bool enable){if(enable){core=MNN::Arm82Functions::get();if(!core||!core->MNNSelectUnaryFunctionForFloat)throw std::runtime_error("Missing ARM82 unary selector");original=core->MNNSelectUnaryFunctionForFloat;calls=0;core->MNNSelectUnaryFunctionForFloat=select;}}
  void restore(){if(core){core->MNNSelectUnaryFunctionForFloat=original;core=nullptr;}}
  ~Scope(){restore();}
};
}
