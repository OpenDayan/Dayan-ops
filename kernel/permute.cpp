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

static inline void config_dims_and_strides_permute(int bs, int h, int n, int m){
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(m);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(m);
}

__device__ void permute_f32_2_1(void *a, void *res, int bs, int h, int n, int m){
    // ppermute.axis0 = 0, ppermute.axis1 = 2, ppermute.axis2 == 1, ppermute.axis3 = 3
    config_dims_and_strides_permute(bs,h,n,m);
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                // Copy a[i][j][k][:] to res[i][k][j][:]
                uint64_t a_idx = i * h * n * m + j * n * m + k * m;
                uint64_t res_idx = i * h * n * m + k * h * m + j * m;

                //这句memcpy也可以
                // memcpy((void *)((long)res + res_idx * sizeof(float)),(void *)((long)a + a_idx * sizeof(float)), m * sizeof(float));

                //或load/store
                __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_idx * sizeof(float)), (void*)nram0);
                __tx_fence();
                __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram0, (void *)((long)res + res_idx * sizeof(float)));
                __tx_fence();
            }
        }
    }
}