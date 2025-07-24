#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const uint64_t nram0 = 0x1000000;
const uint64_t nram1 = 0x2000000;
const uint64_t nram2 = 0x3000000;

static inline void config_dims_and_strides_dup_cont(int a_bs, int a_h, int a_n, int a_m, int res_bs, int res_h, int res_n, int res_m) {

    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(a_m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(a_n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(a_m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(a_h);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(a_m * a_n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(a_bs);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(a_m * a_n * a_h);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(res_m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(res_n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(res_m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(res_h);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(res_m * res_n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(res_bs);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(res_m * res_n * res_h);
}

static inline void config_dims_and_strides_dup_no_cont(){
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
}

__device__ void dup_cont_f32_cont_f32(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m){
    if((a_bs*a_h*a_n*a_m)!=(res_bs*res_h*res_n*res_m)){
        printf("[ERROR] Contiguous fp32 and fp32 tensor has different data num\n");
        return;
    }
    // 直接使用memcpy
    memcpy(res, a, (a_bs * a_h * a_n * a_m * sizeof(float)));

    // 或使用load/store
    // config_dims_and_strides_dup_cont(a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m);
    // __tx_async_nram_load<aTCSR, aTCSR, 2>(a, (void*)nram0);
    // __tx_fence();
    // __tx_async_nram_store<aTCSR, cTCSR, 2>((void*)nram0, res);
    // __tx_fence();
}

__device__ void dup_cont_f32_cont_bf16(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m){
    if((a_bs*a_h*a_n*a_m)!=(res_bs*res_h*res_n*res_m)){
        printf("[ERROR] Contiguous fp32 and bf16 tensor has different data num\n");
        return;
    }
    config_dims_and_strides_dup_cont(a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m);
    __tx_async_nram_load<aTCSR, aTCSR, 2>(a, (void*)nram0);
    __tx_fence();
    __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)nram0, (void *)nram1);
    __tx_fence();
    __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram1, res);
    __tx_fence();
}

__device__ void dup_cont_f32_no_cont_bf16(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m,
                                    int a_nb3, int a_nb2, int a_nb1, int a_nb0, 
                                    int res_nb3, int res_nb2, int res_nb1, int res_nb0){
    if((a_bs*a_h*a_n*a_m)!=(res_bs*res_h*res_n*res_m)){
        printf("[ERROR] Contiguous fp32 and non-contiguous bf16 tensor has different data num\n");
        return;
    }
    config_dims_and_strides_dup_no_cont();
    for (int i3 = 0; i3 < a_bs; ++i3) {
        for (int i2 = 0; i2 < a_h; ++i2) {
            for (int i1 = 0; i1 < a_n; ++i1) {
                for (int i0 = 0; i0 < a_m; ++i0) {
                    uint64_t res_offset = i0 * res_nb0 + i1 * res_nb1 + i2 * res_nb2 + i3 * res_nb3;
                    size_t a_idx = (((i3 * a_h + i2) * a_n + i1) * a_m) + i0;
                    __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_idx * sizeof(float)), (void*)nram0);
                    __tx_fence();
                    __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)nram0, (void *)nram1);
                    __tx_fence();
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram1, (void *)((long)res + res_offset));
                    __tx_fence();
                }
            }
        }
    }
}

__device__ void dup_no_cont_f32_no_cont_bf16(void *a, void *res,
                                    int a_bs, int a_h, int a_n, int a_m,
                                    int res_bs, int res_h, int res_n, int res_m,
                                    int a_nb3, int a_nb2, int a_nb1, int a_nb0, 
                                    int res_nb3, int res_nb2, int res_nb1, int res_nb0){
    if((a_bs*a_h*a_n*a_m)!=(res_bs*res_h*res_n*res_m)){
        printf("[ERROR] Non-contiguous fp32 and non-contiguous fp32 tensor has different data num\n");
        return;
    }
    config_dims_and_strides_dup_no_cont();
    for (int i3 = 0; i3 < a_bs; ++i3) {
        for (int i2 = 0; i2 < a_h; ++i2) {
            for (int i1 = 0; i1 < a_n; ++i1) {
                for (int i0 = 0; i0 < a_m; ++i0) {
                    uint64_t res_offset = i0 * res_nb0 + i1 * res_nb1 + i2 * res_nb2 + i3 * res_nb3;
                    uint64_t a_offset = i0 * a_nb0 + i1 * a_nb1 + i2 * a_nb2 + i3 * a_nb3;
                    __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_offset), (void*)nram0);
                    __tx_fence();
                    __tx_convert_fp32_bf16_round<aTCSR, aTCSR>((void *)nram0, (void *)nram1);
                    __tx_fence();
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram1, (void *)((long)res + res_offset));
                    __tx_fence();
                }
            }
        }
    }
}