#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const uint64_t b_buffer_dram = 0x87000000;
const uint64_t c_buffer_dram = 0x87100000;

static inline void config_dims_and_strides_get_rows(int h1, int n1, int m1, int m0){
    const long aTCFG = 0;
    const long bTCFG = 1;
    const long cTCFG = 2;
    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

    // deal with using tensor later, all set to 1
    // src tensor src0
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);

    // index tensor src1
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE2>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);

    // dst tensor
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);

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

    const long aTCSR = 0;
    const long bTCSR = 1;
    const long cTCSR = 2;
    void *bidx = 0;
    const uint64_t nram0 = 0x1000000;
    const uint64_t nram1 = 0x2000000;
    const uint64_t nram2 = 0x3000000;
    bool print_debug_info = false;
    for (int j = 0; j < h1; ++j) {
        if(print_debug_info) printf("[DEBUG] Outer loop j=%d/%d\n", j, h1);
        for (int k = 0; k < n1; ++k) {
            if(print_debug_info) printf("  [DEBUG] Middle loop k=%d/%d\n", k, n1);
            for (int p = 0; p < m1; ++p) {
                if(print_debug_info) printf("    [DEBUG] Inner loop p=%d/%d\n", p, m1);
                int b_offset = j * n1 * m1 + k * m1 + p;

                // Use the data from src1 to determine the row index to read from src0.
                if(print_debug_info) printf("     [DEBUG] b_offset=%d, addr=0x%lx\n", b_offset, (long)b + b_offset * sizeof(int));
                // __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)b + b_offset * sizeof(int)), (void *)nram0);
                __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)b + b_offset * sizeof(int)), bidx);
                __tx_fence();
                __tx_async_nram_store<bTCSR, bTCSR, 2>(bidx, (void *)((long)b_buffer_dram));
                __tx_fence();
                memcpy(&b_buffer,(void *)((long)b_buffer_dram),sizeof(int));
                int row_idx = b_buffer;
                if(print_debug_info) printf("[DEBUG]      Read row_idx = %d (0x%x)\n", row_idx, row_idx);
                if (row_idx < 0 || row_idx > n0) {
                    printf("[ERROR] invalid row_idx value: %d",row_idx);
                    return 0;
                }
                // Read all m0 elements from the row indexed by row_idx in src0.
                for (int l = 0; l < m0; ++l) {
                    int a_offset = j * n1 * n0 * m0 + k * n0 * m0 + row_idx * m0 + l;
                    int c_offset = j * n1 * m1 * m0 + k * m1 * m0 + p * m0 + l;
                    if(print_debug_info) printf("        [DEBUG] l=%d, a_offset=%d, c_offset=%d\n", l, a_offset, c_offset);
                    uint64_t* f32_val;
                    // use nram1 to move elem
                    __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_offset * sizeof(float)), (void *)nram1);
                    __tx_fence();
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram1, (void *)((long)c + c_offset * sizeof(float)));
                    __tx_fence();
                }
            }
        }
    }
    // printf("End device get_rows_f32_i32_f32\n");
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

    const long aTCSR = 0;
    const long bTCSR = 1;
    const long cTCSR = 2;
    void *bidx = 0;
    void *temp = (void *)((long)bidx + sizeof(float));;
    const uint64_t nram0 = 0x1000000;
    const uint64_t nram1 = 0x2000000;
    const uint64_t nram2 = 0x3000000;

    bool print_debug_info = false;
    for (int j = 0; j < h1; ++j) {
        if(print_debug_info) printf("[DEBUG] Outer loop j=%d/%d\n", j, h1);
        for (int k = 0; k < n1; ++k) {
            if(print_debug_info) printf("  [DEBUG] Middle loop k=%d/%d\n", k, n1);
            for (int p = 0; p < m1; ++p) {
                if(print_debug_info) printf("    [DEBUG] Inner loop p=%d/%d\n", p, m1);
                int b_offset = j * n1 * m1 + k * m1 + p;

                // Use the data from src1 to determine the row index to read from src0.
                if(print_debug_info) printf("     [DEBUG] b_offset=%d, addr=0x%lx\n", b_offset, (long)b + b_offset * sizeof(int));
                __tx_async_nram_load<bTCSR, bTCSR, 2>((void *)((long)b + b_offset * sizeof(int)), bidx);
                __tx_fence();
                __tx_async_nram_store<bTCSR, bTCSR, 2>(bidx, (void *)((long)b_buffer_dram));
                __tx_fence();
                memcpy(&b_buffer,(void *)((long)b_buffer_dram),sizeof(int));
                int row_idx = b_buffer;
                if(print_debug_info) printf("[DEBUG]      Read row_idx = %d (0x%x)\n", row_idx, row_idx);
                if (row_idx < 0 || row_idx > n0) {
                printf("[ERROR] invalid row_idx value: %d",row_idx);
                return 0;
                }
                // Read all m0 elements from the row indexed by row_idx in src0.
                for (int l = 0; l < m0; ++l) {
                    int a_offset = j * n1 * n0 * m0 + k * n0 * m0 + row_idx * m0 + l;
                    int c_offset = j * n1 * m1 * m0 + k * m1 * m0 + p * m0 + l;
                    if(print_debug_info) printf("        [DEBUG] l=%d, a_offset=%d, c_offset=%d\n", l, a_offset, c_offset);
                    uint64_t* f32_val;
                    // get src0 element with nram0
                    __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_offset * sizeof(uint16_t)), (void *)nram0);
                    __tx_fence();
                    __tx_convert_bf16_fp32_round<0, 1>((void *)nram0, (void *)nram1);
                    __tx_fence();
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram1, (void *)((long)c + c_offset * sizeof(float)));
                    __tx_fence();
                    
                    // This is converting bf16 to fp32 manually.
                    // __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)a + a_offset * sizeof(uint16_t)), (void *)nram0);
                    // __tx_fence();
                    // __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram0, (void *)((long)c_buffer_dram));
                    // __tx_fence();
                    // memcpy(&c_buffer_fp16,(void *)((long)c_buffer_dram),sizeof(uint16_t));
                    // // printf("[DEBUG] DRAM Copy BF16 = 0x%04x\n", c_buffer_fp16);
                    // c_buffer_u32 = ((uint32_t)c_buffer_fp16) << 16;
                    // // c_buffer_u32 = ((uint32_t)c_buffer_fp16) << 0;
                    // // printf("[DEBUG] Convert to FP32 u32 = 0x%08x\n", c_buffer_u32);
                    // memcpy((void *)((long)c_buffer_dram), &c_buffer_u32, sizeof(uint32_t));
                    // __tx_async_nram_load<aTCSR, aTCSR, 2>((void *)((long)c_buffer_dram), (void *)nram0);
                    // __tx_fence();
                    // __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram0, (void *)((long)c + c_offset * sizeof(float)));
                    // __tx_fence();
                }
            }
        }
    }
    // printf("End device get_rows_bf16_i32_f32\n");
    return 0;
}
