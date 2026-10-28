#include "MnnGraph.h"
// Build verifies the optional MNN interface, without loading random graphs.
static_assert(sizeof(float)==4,"Requires FP32 host IO");
