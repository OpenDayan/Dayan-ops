#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const uint64_t b_buffer_dram = 0x87000000;
const uint64_t c_buffer_dram = 0x87100000;
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const uint64_t nram0 = 0x1000000;
const uint64_t nram1 = 0x2000000;
const uint64_t nram2 = 0x3000000;
static inline void config_dims_and_strides_get_rows(int h1, int n1, int m1, int m0){
    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

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

    // index tensor src1
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(1);

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

static int b_buffer;
static float c_buffer;
static uint16_t c_buffer_fp16;
static uint32_t c_buffer_u32;
__device__ long get_rows_f32_i32_f32(void *a, void *b, void *c, 
                                    int h1, int n1, int m1, int m0, int n0){

    // assume the shape fits
    //a: source data, b: index c: result dst
    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)
    config_dims_and_strides_get_rows(h1, n1, m1, m0);

    for (int j = 0; j < h1; ++j) {
        for (int k = 0; k < n1; ++k) {
            for (int p = 0; p < m1; ++p) {
                int b_offset = j * n1 * m1 + k * m1 + p;

                // Use the data from src1 to determine the row index to read from src0.
                __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)b + b_offset * sizeof(int)), (void *)nram0);
                // __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)b + b_offset * sizeof(int)), bidx);
                __tx_fence();
                __tx_async_nram_store<bTCSR, bTCSR, 2>((void*)nram0, (void *)((long)b_buffer_dram));
                __tx_fence();
                memcpy(&b_buffer,(void *)((long)b_buffer_dram),sizeof(int));
                int row_idx = b_buffer;
                if (row_idx < 0 || row_idx > n0) {
                    printf("[ERROR] invalid row_idx value: %d",row_idx);
                    return 0;
                }
                int a_offset = j * n1 * n0 * m0 + k * n0 * m0 + row_idx * m0;
                int c_offset = j * n1 * m1 * m0 + k * m1 * m0 + p * m0;
                __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_offset * sizeof(float)), (void *)nram1);
                __tx_fence();
                __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram1, (void *)((long)c + c_offset * sizeof(float)));
                __tx_fence();
            }
        }
    }
    return 0;
}
__device__ long get_rows_bf16_i32_f32(void *a, void *b, void *c, 
                                    int h1, int n1, int m1, int m0, int n0){

    // assume the shape fits
    //a: source data, b: index c: result dst
    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

    config_dims_and_strides_get_rows(h1, n1, m1, m0);

    for (int j = 0; j < h1; ++j) {
        for (int k = 0; k < n1; ++k) {
            for (int p = 0; p < m1; ++p) {
                int b_offset = j * n1 * m1 + k * m1 + p;
                // Use the data from src1 to determine the row index to read from src0.
                __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)b + b_offset * sizeof(int)), (void *)nram0);
                __tx_fence();
                __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram0, (void *)((long)b_buffer_dram));
                __tx_fence();
                memcpy(&b_buffer,(void *)((long)b_buffer_dram),sizeof(int));
                int row_idx = b_buffer;
                if (row_idx < 0 || row_idx > n0) {
                printf("[ERROR] invalid row_idx value: %d",row_idx);
                return 0;
                }
                // Read all m0 elements from the row indexed by row_idx in src0.
                int a_offset = j * n1 * n0 * m0 + k * n0 * m0 + row_idx * m0;
                int c_offset = j * n1 * m1 * m0 + k * m1 * m0 + p * m0;
                // get src0 element with nram0
                __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_offset * sizeof(uint16_t)), (void *)nram0);
                __tx_fence();
                __tx_convert_bf16_fp32_round<aTCSR, cTCSR>((void *)nram0, (void *)nram1);
                __tx_fence();
                __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram1, (void *)((long)c + c_offset * sizeof(float)));
                __tx_fence();
                
                // This is converting bf16 to fp32 manually.
                // for (int l = 0; l < m0; ++l) {
                //     a_offset = j * n1 * n0 * m0 + k * n0 * m0 + row_idx * m0 + l;
                //     c_offset = j * n1 * m1 * m0 + k * m1 * m0 + p * m0 + l;
                // // Using bTCSR here is for move 1 element every time as a and c are set to m0
                // __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)a + a_offset * sizeof(uint16_t)), (void *)nram0);
                // __tx_fence();
                // __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram0, (void *)((long)c_buffer_dram));
                // __tx_fence();
                // memcpy(&c_buffer_fp16,(void *)((long)c_buffer_dram),sizeof(uint16_t));
                // c_buffer_u32 = ((uint32_t)c_buffer_fp16) << 16;
                // memcpy((void *)((long)c_buffer_dram), &c_buffer_u32, sizeof(uint32_t));
                // __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)c_buffer_dram), (void *)nram0);
                // __tx_fence();
                // __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram0, (void *)((long)c + c_offset * sizeof(float)));
                // __tx_fence();}
            }
        }
    }
    return 0;
}
