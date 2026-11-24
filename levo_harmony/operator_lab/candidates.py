"""Auditable deterministic candidate generation; never mutate pinned MNN sources."""
import hashlib


def mulsilu_tile_source(original,tile):
    if not isinstance(tile,int) or tile not in {0,256,1024,4096,8192}:raise ValueError('Unregistered tile recipe')
    original_hash=hashlib.sha256(original.encode()).hexdigest()
    includes=['CPUBinary.hpp','CPUBinaryInt8.hpp','CPUBackend.hpp','compute/CommonOptFunction.h','compute/ConvOpt.h','BinaryUtils.hpp']
    text=original
    for name in includes:
        old='#include "'+name+'"'
        if text.count(old)!=1:raise ValueError('Pinned CPUBinary include changed: '+name)
        text=text.replace(old,'#include "backend/cpu/'+name+'"')
    anchor='namespace MNN {'
    if text.count(anchor)!=1:raise ValueError('Pinned namespace changed')
    support=f'''#include <atomic>
#include <cstdint>
namespace LevoMulSiluTile {{ static std::atomic<uint64_t> calls{{0}}; constexpr int tile={tile}; }}
extern "C" void LlmOperatorCandidateSetup() {{}}
extern "C" uint64_t LlmOperatorCandidateCalls() {{ return LevoMulSiluTile::calls.load(); }}
'''
    text=text.replace(anchor,support+'\n'+anchor)
    anchor='auto i1 = input1->host<int8_t>();'
    if text.count(anchor)!=1:raise ValueError('Pinned MUL_SILU storage changed')
    text=text.replace(anchor,anchor+'''
        const uintptr_t begin=reinterpret_cast<uintptr_t>(o0), a=reinterpret_cast<uintptr_t>(i0), b=reinterpret_cast<uintptr_t>(i1);
        const uintptr_t span=uintptr_t(size)*bytes;
        const bool independent=(begin+span<=a||a+span<=begin)&&(begin+span<=b||b+span<=begin);
        const bool eligible=bytes==4&&independent;
        if(eligible&&(LevoMulSiluTile::tile==0||size>LevoMulSiluTile::tile*schedule.second))LevoMulSiluTile::calls.fetch_add(1,std::memory_order_relaxed);
''')
    old='''mSilu((float*)out, (float*)inp1, realSize);
                mMul((float*)out, (float*)out, (float*)inp, realSize, -1);'''
    if text.count(old)!=1:raise ValueError('Pinned MUL_SILU arithmetic changed')
    tiled='''if(eligible && LevoMulSiluTile::tile>0 && realSize>LevoMulSiluTile::tile) {
                    for(int offset=0;offset<realSize;offset+=LevoMulSiluTile::tile) {
                        const int count=std::min(LevoMulSiluTile::tile,realSize-offset);
                        mSilu(reinterpret_cast<float*>(out)+offset,reinterpret_cast<float*>(inp1)+offset,count);
                        mMul(reinterpret_cast<float*>(out)+offset,reinterpret_cast<float*>(out)+offset,reinterpret_cast<float*>(inp)+offset,count,-1);
                    }
                } else {
                    '''+old+'''
                }'''
    text=text.replace(old,tiled)
    return '// Generated isolated CPUBinary recipe; normalized source SHA256 '+original_hash+'\n'+text
