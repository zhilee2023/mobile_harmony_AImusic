#pragma once
// Numerical sensitivity research only: FP16 outputs carry dequantized values.
// This is not packed-A4 storage or an INT4 convolution speed implementation.
namespace LevoVaeActivationResearch {
inline int bits=0;
inline bool rotate=false;
inline size_t minimumArea=0;
inline std::atomic<uint64_t> calls{0},elements{0};
inline bool selected(bool packed,int channels,size_t area){return bits&&packed&&channels%32==0&&area>=minimumArea;}
inline void hadamard(float32x4_t* v){
    for(int i=0;i<8;++i){
        auto even=vuzp1q_f32(v[i],v[i]),odd=vuzp2q_f32(v[i],v[i]);
        auto pairs=vzip1q_f32(vaddq_f32(even,odd),vsubq_f32(even,odd));
        v[i]=vcombine_f32(vadd_f32(vget_low_f32(pairs),vget_high_f32(pairs)),vsub_f32(vget_low_f32(pairs),vget_high_f32(pairs)));
    }
    for(int stride=1;stride<8;stride*=2)for(int base=0;base<8;base+=2*stride)for(int j=0;j<stride;++j){
        auto a=v[base+j],b=v[base+j+stride];v[base+j]=vaddq_f32(a,b);v[base+j+stride]=vsubq_f32(a,b);
    }
}
inline void qdq(float16x8_t* half){
    float32x4_t values[8],signs[8];
    alignas(16) static const float sign[32]={1,-1,-1,1, -1,1,1,1, 1,-1,-1,-1, 1,1,-1,1,
                                            -1,1,-1,-1, 1,-1,1,1, -1,-1,1,-1, 1,1,-1,-1};
    for(int i=0;i<4;++i){values[2*i]=vcvt_f32_f16(vget_low_f16(half[i]));values[2*i+1]=vcvt_f32_f16(vget_high_f16(half[i]));}
    if(rotate){for(int i=0;i<8;++i){signs[i]=vld1q_f32(sign+4*i);values[i]=vmulq_f32(values[i],signs[i]);}hadamard(values);}
    auto maximum=vdupq_n_f32(0);for(auto v:values)maximum=vmaxq_f32(maximum,vabsq_f32(v));const float peak=vmaxvq_f32(maximum);
    if(peak>0&&std::isfinite(peak)){
        const float limit=bits==8?127.f:7.f;
        for(auto& v:values)v=vmulq_n_f32(vrndnq_f32(vmulq_n_f32(v,limit/peak)),peak/limit);
    }
    if(rotate){hadamard(values);for(int i=0;i<8;++i)values[i]=vmulq_n_f32(vmulq_f32(values[i],signs[i]),1.f/32);}
    for(int i=0;i<4;++i)half[i]=vcombine_f16(vcvt_f16_f32(values[2*i]),vcvt_f16_f32(values[2*i+1]));
}
inline void execute(MNN::CPUBackend* cpu,const float16_t* x,float16_t* y,const float16_t* alpha,const float16_t* beta,int channels,size_t area){
    constexpr size_t tile=2048;const size_t slices=(area+tile-1)/tile;
    MNN::ThreadPool::TASK task;task.second=cpu->threadNumber();
    task.first=[&](int worker){
        for(size_t job=worker;job<size_t(channels/32)*slices;job+=task.second){
            const size_t row=job/slices,start=(job%slices)*tile,end=std::min(area,start+tile);float16x8_t a[4],b[4];
            for(int c=0;c<4;++c){a[c]=vld1q_f16(alpha+row*32+c*8);b[c]=vld1q_f16(beta+row*32+c*8);}
            for(size_t t=start;t<end;++t){float16x8_t values[4];
                for(int c=0;c<4;++c)values[c]=LevoVaeSnake::snake8(vld1q_f16(x+(row*4+c)*area*8+t*8),a[c],b[c]);
                qdq(values);
                for(int c=0;c<4;++c)vst1q_f16(y+(row*4+c)*area*8+t*8,values[c]);
            }
        }
    };
    cpu->enqueue(task);++calls;elements+=size_t(channels)*area;
}
}
