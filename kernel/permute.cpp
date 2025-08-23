#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const uint64_t uram_a = 0x0;

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
        int i_stride = h * n * m;
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                // Copy a[i][j][k][:] to res[i][k][j][:]
                uint64_t a_idx = i * i_stride + j * n * m + k * m;
                uint64_t res_idx = i * i_stride + k * h * m + j * m;
                __tx_async_uram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_idx * sizeof(float)), (void*)uram_a);
                __tx_fence();
                __tx_async_uram_store<cTCSR, cTCSR, 2>((void*)uram_a, (void *)((long)res + res_idx * sizeof(float)));
                __tx_fence();
            }
        }
    }
}