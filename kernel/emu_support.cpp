#include "npu_instruction.hpp"

#ifdef __EMULATED_RVT_EXT__
RVTEnv* __rvt_env;
static RVTEnv __real_env;
static char __real_nram [NRAM_SIZE];
static long __real_tcsr [TCSR_SIZE];

extern "C" void __tx_init() {
    __rvt_env = &__real_env;
    __rvt_env->nram = __real_nram;
    __rvt_env->tcsr = __real_tcsr;
    __rvt_env->tcfg = (TCFG*)__real_tcsr;
}
#else
extern "C" void __tx_init() {
    // Do nothing
}
#endif
