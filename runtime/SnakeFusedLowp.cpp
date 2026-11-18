// Link only into targets that explicitly enable the verified Snake operator.
// All ordinary MNN operators retain their original precision dispatch.
#include "include/levo/kernels/SnakeFusedOp.hpp"
extern "C" bool __real__ZN3MNN13OpCommonUtils20opCompabilityForLowpEPKNS_2OpEi(const MNN::Op*, int);
extern "C" bool __wrap__ZN3MNN13OpCommonUtils20opCompabilityForLowpEPKNS_2OpEi(const MNN::Op* op, int bytes) {
    if (LevoSnakeFused::identifies(op)) return bytes == 2;
    return __real__ZN3MNN13OpCommonUtils20opCompabilityForLowpEPKNS_2OpEi(op, bytes);
}
