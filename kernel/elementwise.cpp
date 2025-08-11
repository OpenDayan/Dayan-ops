#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

// 向量化处理的块大小
#define MTile 1024  // 每次处理1024个float元素

const uint64_t nram0 = 0x0;        // A tensor: 0x0 - 0x14000 (80KB)
const uint64_t nram1 = 0x14000;    // B tensor: 0x14000 - 0x28000 (80KB)  
const uint64_t nram2 = 0x28000;    // C tensor: 0x28000 - 0x3C000 (80KB)
const int aTCSR = 0;
const int bTCSR = 1; 
const int cTCSR = 2;

template<int TCSR_ID>
static inline void config_tcsr_vector(int m) {
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE2>(m);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE3>(m);
}

// 广播计算
static inline uint8_t get_broadcast(int a_dim0, int a_dim1, int a_dim2, int a_dim3,
                                   int b_dim0, int b_dim1, int b_dim2, int b_dim3) {
    uint8_t broadcast = 0;

    if (a_dim0 != b_dim0 && (a_dim0 == 1 || b_dim0 == 1)) {
        broadcast |= 0b0001;
    }

    if (a_dim1 != b_dim1 && (a_dim1 == 1 || b_dim1 == 1)) {
        broadcast |= 0b0010;
    }

    if (a_dim2 != b_dim2 && (a_dim2 == 1 || b_dim2 == 1)) {
        broadcast |= 0b0100;
    }

    if (a_dim3 != b_dim3 && (a_dim3 == 1 || b_dim3 == 1)) {
        broadcast |= 0b1000;
    }

    return broadcast;
}

__device__ long kernel_add_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{
    // 计算广播参数
    uint8_t broadcast = get_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                     psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);

    // printf("[KERNEL] Broadcasting pattern: 0x%x\n", broadcast);
        
    int a_stride_dim1 = psrc0_ne0;
    int a_stride_dim2 = psrc0_ne0 * psrc0_ne1;
    int a_stride_dim3 = psrc0_ne0 * psrc0_ne1 * psrc0_ne2;
    
    int b_stride_dim1 = psrc1_ne0;
    int b_stride_dim2 = psrc1_ne0 * psrc1_ne1;
    int b_stride_dim3 = psrc1_ne0 * psrc1_ne1 * psrc1_ne2;

    for (int i = 0; i < psrc0_ne3; ++i) {
        int i1 = (psrc1_ne3 == 1) ? 0 : i;
        
        for (int j = 0; j < psrc0_ne2; ++j) {
            int j1 = (psrc1_ne2 == 1) ? 0 : j;
            
            for (int k = 0; k < psrc0_ne1; ++k) {
                int k1 = (psrc1_ne1 == 1) ? 0 : k;

                // 向量化处理最内层维度
                int remaining_elements = psrc0_ne0;
                int processed_in_row = 0;
                
                while (remaining_elements > 0) {
                    int current_chunk = (remaining_elements > MTile) ? 
                                        MTile : remaining_elements;
                    
                    config_tcsr_vector<aTCSR>(current_chunk);
                    config_tcsr_vector<bTCSR>(current_chunk);
                    config_tcsr_vector<cTCSR>(current_chunk);

                    // 计算偏移
                    long a_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + processed_in_row;
                    float* a_elem_ptr = (float*)a_ptr + a_offset;
                    long b_offset;
                    if (psrc1_ne0 == 1) {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    } else {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + processed_in_row;
                    }
                    float* b_elem_ptr = (float*)b_ptr + b_offset;
                    long c_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + processed_in_row;
                    float* c_elem_ptr = (float*)c_ptr + c_offset;

                    __tx_async_nram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)nram0);
                    if (psrc1_ne0 == 1) {
                        config_tcsr_vector<bTCSR>(1);
                        __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
                    } else {
                        __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
                    }
                    __tx_fence();

                    uint8_t current_broadcast = (psrc1_ne0 == 1) ? (broadcast | 0b0001) : broadcast;
                        
#define CALL_TX_FP_ADD_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)nram0, (void *)nram1, (void *)nram2); \
    break;

                    switch (current_broadcast) {
                        CALL_TX_FP_ADD_FP32(0b0000)
                        CALL_TX_FP_ADD_FP32(0b0001)
                        CALL_TX_FP_ADD_FP32(0b0010)
                        CALL_TX_FP_ADD_FP32(0b0011)
                        CALL_TX_FP_ADD_FP32(0b0100)
                        CALL_TX_FP_ADD_FP32(0b0101)
                        CALL_TX_FP_ADD_FP32(0b0110)
                        CALL_TX_FP_ADD_FP32(0b0111)
                        CALL_TX_FP_ADD_FP32(0b1000)
                        CALL_TX_FP_ADD_FP32(0b1001)
                        CALL_TX_FP_ADD_FP32(0b1010)
                        CALL_TX_FP_ADD_FP32(0b1011)
                        CALL_TX_FP_ADD_FP32(0b1100)
                        CALL_TX_FP_ADD_FP32(0b1101)
                        CALL_TX_FP_ADD_FP32(0b1110)
                        CALL_TX_FP_ADD_FP32(0b1111)
                    default:
                        printf("[KERNEL] ERROR: Unsupported broadcast pattern: %d\n", current_broadcast);
                        return -1;
                    }

#undef CALL_TX_FP_ADD_FP32

                    __tx_fence();
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, c_elem_ptr);
                    __tx_fence();
                    
                    processed_in_row += current_chunk;
                    remaining_elements -= current_chunk;
                }
            }
        }
    }

    // printf("[KERNEL] NPU kernel_add_fp32 execution completed successfully!\n");
    return 0;
}

__device__ long kernel_mul_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{
    // 计算广播参数
    uint8_t broadcast = get_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                     psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);

    // printf("[KERNEL] Broadcasting pattern: 0x%x\n", broadcast);
        
    int a_stride_dim1 = psrc0_ne0;
    int a_stride_dim2 = psrc0_ne0 * psrc0_ne1;
    int a_stride_dim3 = psrc0_ne0 * psrc0_ne1 * psrc0_ne2;
    
    int b_stride_dim1 = psrc1_ne0;
    int b_stride_dim2 = psrc1_ne0 * psrc1_ne1;
    int b_stride_dim3 = psrc1_ne0 * psrc1_ne1 * psrc1_ne2;

    for (int i = 0; i < psrc0_ne3; ++i) {
        int i1 = (psrc1_ne3 == 1) ? 0 : i;
        
        for (int j = 0; j < psrc0_ne2; ++j) {
            int j1 = (psrc1_ne2 == 1) ? 0 : j;
            
            for (int k = 0; k < psrc0_ne1; ++k) {
                int k1 = (psrc1_ne1 == 1) ? 0 : k;
                
                // 向量化处理最内层维度
                int remaining_elements = psrc0_ne0;
                int processed_in_row = 0;
                
                while (remaining_elements > 0) {
                    int current_chunk = (remaining_elements > MTile) ? 
                                        MTile : remaining_elements;
                    
                    config_tcsr_vector<aTCSR>(current_chunk);
                    config_tcsr_vector<bTCSR>(current_chunk);
                    config_tcsr_vector<cTCSR>(current_chunk);
                    
                    // 计算偏移
                    long a_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + processed_in_row;
                    float* a_elem_ptr = (float*)a_ptr + a_offset;
                    long b_offset;
                    if (psrc1_ne0 == 1) {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    } else {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + processed_in_row;
                    }
                    float* b_elem_ptr = (float*)b_ptr + b_offset;
                    long c_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + processed_in_row;
                    float* c_elem_ptr = (float*)c_ptr + c_offset;

                    __tx_async_nram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)nram0);
                    if (psrc1_ne0 == 1) {
                        config_tcsr_vector<bTCSR>(1);
                        __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
                    } else {
                        __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
                    }
                    __tx_fence();

                    uint8_t current_broadcast = (psrc1_ne0 == 1) ? (broadcast | 0b0001) : broadcast;
                    
#define CALL_TX_FP_MUL_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)nram0, (void *)nram1, (void *)nram2); \
    break;

                    switch (current_broadcast) {
                        CALL_TX_FP_MUL_FP32(0b0000)
                        CALL_TX_FP_MUL_FP32(0b0001)
                        CALL_TX_FP_MUL_FP32(0b0010)
                        CALL_TX_FP_MUL_FP32(0b0011)
                        CALL_TX_FP_MUL_FP32(0b0100)
                        CALL_TX_FP_MUL_FP32(0b0101)
                        CALL_TX_FP_MUL_FP32(0b0110)
                        CALL_TX_FP_MUL_FP32(0b0111)
                        CALL_TX_FP_MUL_FP32(0b1000)
                        CALL_TX_FP_MUL_FP32(0b1001)
                        CALL_TX_FP_MUL_FP32(0b1010)
                        CALL_TX_FP_MUL_FP32(0b1011)
                        CALL_TX_FP_MUL_FP32(0b1100)
                        CALL_TX_FP_MUL_FP32(0b1101)
                        CALL_TX_FP_MUL_FP32(0b1110)
                        CALL_TX_FP_MUL_FP32(0b1111)
                    default:
                        printf("[KERNEL] ERROR: Unsupported broadcast pattern: %d\n", current_broadcast);
                        return -1;
                    }

#undef CALL_TX_FP_MUL_FP32

                    __tx_fence();

                    // 存储结果
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, c_elem_ptr);
                    __tx_fence();
                    
                    processed_in_row += current_chunk;
                    remaining_elements -= current_chunk;
                }
            }
        }
    }

    // printf("[KERNEL] NPU kernel_mul_fp32 execution completed successfully!\n");
    return 0;
}