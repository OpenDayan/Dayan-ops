#include "npu_instruction.hpp"
#include "npu_kernel.h"
// #include <simsimd/simsimd.h>

#ifdef __EMULATED_RVT_EXT__
RVTEnv* __rvt_env;
static RVTEnv __real_env;
static char __real_nram [NRAM_SIZE];
static long __real_tcsr [TCSR_SIZE];

__device__ void __tx_init() {
    // simsimd_flush_denormals(); // Optional, to avoid performance penalties on denormal numbers
    __rvt_env = &__real_env;
    __rvt_env->nram = __real_nram;
    __rvt_env->tcsr = __real_tcsr;
    __rvt_env->tcfg = (TCFG*)__real_tcsr;
}
#else
__device__ void __tx_init() {
    // Do nothing
}
#endif
