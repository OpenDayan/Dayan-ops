#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <type_traits>

extern int _uram[256 * 1024];

template <ulong TCFG_ID> static inline void config_tcsr(TCFG *tcsr) {
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_DIM0>(tcsr[0].dim0);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_STRIDE0>(tcsr[0].stride0);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_DIM1>(tcsr[0].dim1);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_STRIDE1>(tcsr[0].stride1);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_DIM2>(tcsr[0].dim2);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_STRIDE2>(tcsr[0].stride2);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_DIM3>(tcsr[0].dim3);
    __tx_tcsr_write<TCFG_ID * 8 + TCSR_STRIDE3>(tcsr[0].stride3);
}

template <typename T, ulong TCFG_A, ulong TCFG_C, typename INST_F,
          typename DATAGENA_F, typename REF_F, typename CHK_F>
static int test_unary_inst(INST_F INST, DATAGENA_F DATAGENA, REF_F REF,
                            CHK_F CHK, ulong dim0, ulong dim1 = 1, ulong dim2 = 1, ulong dim3 = 1) {
    TCFG cfg = {.dim0 = dim0,
                .stride0 = 1,
                .dim1 = dim1,
                .stride1 = dim0,
                .dim2 = dim2,
                .stride2 = dim0 * dim1,
                .dim3 = dim3,
                .stride3 = dim0 * dim1 * dim2};
    config_tcsr<TCFG_A>(&cfg);
    config_tcsr<TCFG_C>(&cfg);
    ulong uram_size = dim0 * dim1 * dim2 * dim3 * sizeof(T);
    ulong uram_size_pad = (uram_size + 31) & ~31;
    T *dev_a = (T *)_uram;
    T *dev_c = dev_a + uram_size_pad;
    // Initialize data in dev_a and dev_b
    for (ulong d3 = 0; d3 < dim3; ++d3) {
        for (ulong d2 = 0; d2 < dim2; ++d2) {
            for (ulong d1 = 0; d1 < dim1; ++d1) {
                for (ulong d0 = 0; d0 < dim0; ++d0) {
                    ulong i = d3 * dim2 * dim1 * dim0 + d2 * dim1 * dim0 + d1 * dim0 + d0;
                    // 支持 DATAGENA/DATAGENB 最多四个参数，使用 if constexpr(SFINAE) 判断参数数量
                    if constexpr (std::is_invocable_v<decltype(DATAGENA), ulong, ulong, ulong,
                                                      ulong>) {
                        dev_a[i] = DATAGENA(d0, d1, d2, d3);
                    } else if constexpr (std::is_invocable_v<decltype(DATAGENA), ulong, ulong,
                                                             ulong>) {
                        dev_a[i] = DATAGENA(d0, d1, d2);
                    } else if constexpr (std::is_invocable_v<decltype(DATAGENA), ulong, ulong>) {
                        dev_a[i] = DATAGENA(d0, d1);
                    } else if constexpr (std::is_invocable_v<decltype(DATAGENA), ulong>) {
                        dev_a[i] = DATAGENA(d0);
                    } else {
                        dev_a[i] = DATAGENA();
                    }
                }
            }
        }
    }
    // Run the instruction
    INST(dev_a, dev_c);
    // Barrier
    __tx_sb_bar();

    for (ulong d3 = 0; d3 < dim3; ++d3) {
        for (ulong d2 = 0; d2 < dim2; ++d2) {
            for (ulong d1 = 0; d1 < dim1; ++d1) {
                for (ulong d0 = 0; d0 < dim0; ++d0) {
                    ulong i = d3 * dim2 * dim1 * dim0 + d2 * dim1 * dim0 + d1 * dim0 + d0;
                    T expected = REF(dev_a[i]); // 计算参考值
                    T actual = dev_c[i];
                    if (!CHK(expected, actual)) {
                        printf("Error at %lu, %lu, %lu, %lu\n", d0, d1, d2, d3);
                        printf("a = %lu, c = %lu\n", dev_a[i], dev_c[i]);
                        printf("expected = 0x%lx, actual = 0x%lx\n", expected, actual);
                        return -1;
                    }
                }
            }
        }
    }
    return 0;
}

// CONVERT
__device__ long test_legacy_convert_fp32_bf16_inst(void) {
    printf("[ERROR] test_legacy_convert_fp32_bf16_inst not implemented"); // TODO
    return -1;
}

__device__ long test_legacy_convert_fp32_e4m3_inst(void) {
    printf("[ERROR] test_legacy_convert_fp32_e4m3_inst not implemented"); // TODO
    return -1;
}

// SFU
__device__ long test_legacy_recip_fp32_inst(void) {
    printf("[ERROR] test_legacy_recip_fp32_inst not implemented"); // TODO
    return -1;
}

__device__ long test_legacy_relu_fp32_inst(void) {
    printf("[ERROR] test_legacy_relu_fp32_inst not implemented"); // TODO
    return -1;
}

__device__ long test_legacy_rsqrt_fp32_inst(void) {
    printf("[ERROR] test_legacy_rsqrt_fp32_inst not implemented"); // TODO
    return -1;
}

__device__ long test_legacy_sin_fp32_inst(void) {
    printf("[ERROR] test_legacy_sin_fp32_inst not implemented"); // TODO
    return -1;
}

__device__ long test_legacy_cos_fp32_inst(void) {
    return test_unary_inst<float, 0, 0>(
        __tx_sfu_sin_fp32<0, 0>,
        [](ulong d0, ulong d1, ulong d2, ulong d3) { return (float)(d0 + d1); },
        [](auto a) -> float { return (float)cosf((float)a); },
        [](auto expected, auto actual) { return fabs(expected - actual) < 1e-5f; },
        16);
}

__device__ long test_legacy_exp_fp32_inst(void) {
    printf("[ERROR] test_legacy_exp_fp32_inst not implemented"); // TODO
    return -1;
}
