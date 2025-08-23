#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const uint64_t uram_a = 0x0;

static inline void config_dims_and_strides_get_rows(int m0){
    // move m0 elements every time
    // src tensor src0
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(m0);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(m0);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(m0);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(m0);

    // dst tensor
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(m0);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(m0);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(m0);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(m0);
}

__device__ long get_rows_f32_i32_f32(void *a, void *b, void *c, 
                                    int h1, int n1, int m1, int m0, int n0){
    // assume the shape fits
    //a: source data, b: index c: result dst
    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)
    config_dims_and_strides_get_rows(m0);
    // 实际trace中只有b只有一个数字，优化掉外面两层循环, m1 = 1
    for (int p = 0; p < m1; ++p) {
        int *b_int = (int*)b;
        int row_idx = b_int[p];
        if (row_idx < 0 || row_idx > n0) {
            printf("[ERROR] invalid row_idx value: %d",row_idx);
            return 0;
        }
        __tx_async_uram_load<aTCSR, aTCSR, 2>((void *)((long)a + row_idx * m0 * sizeof(float)), (void *)uram_a);
        __tx_fence();
        __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram_a, (void *)((long)c + p * m0 * sizeof(float)));
        __tx_fence();
    }
    return 0;
}

