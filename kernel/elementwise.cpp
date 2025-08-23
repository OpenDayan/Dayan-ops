#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

// 向量化处理的块大小
#define MTile 1024  // 每次处理1024个float元素

const uint64_t uram0 = 0x0;        // A tensor: 0x0 - 0x14000 (80KB)
const uint64_t uram1 = 0x14000;    // B tensor: 0x14000 - 0x28000 (80KB)  
const uint64_t uram2 = 0x28000;    // C tensor: 0x28000 - 0x3C000 (80KB)
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

// 检查是否需要广播
static inline bool needs_broadcast(int a_dim0, int a_dim1, int a_dim2, int a_dim3,
                                  int b_dim0, int b_dim1, int b_dim2, int b_dim3) {
    return (a_dim0 != b_dim0 || a_dim1 != b_dim1 || a_dim2 != b_dim2 || a_dim3 != b_dim3);
}

__device__ long kernel_add_fp32_without_broadcast(void *a_ptr, void *b_ptr, void *c_ptr,
                                             int ne0, int ne1, int ne2, int ne3) {
    long total_elements = (long)ne3 * ne2 * ne1 * ne0;
    
    int stride_dim1 = ne0;
    int stride_dim2 = ne0 * ne1;
    int stride_dim3 = ne0 * ne1 * ne2;

    for (int i = 0; i < ne3; ++i) {
        for (int j = 0; j < ne2; ++j) {
            for (int k = 0; k < ne1; ++k) {
                int total_chunks = (ne0 + MTile - 1) / MTile;
                
                if (total_chunks == 1) {
                    int current_chunk = ne0;
                    
                    config_tcsr_vector<aTCSR>(current_chunk);
                    config_tcsr_vector<bTCSR>(current_chunk);
                    config_tcsr_vector<cTCSR>(current_chunk);

                    long offset = i * stride_dim3 + j * stride_dim2 + k * stride_dim1;
                    float* a_elem_ptr = (float*)a_ptr + offset;
                    float* b_elem_ptr = (float*)b_ptr + offset;
                    float* c_elem_ptr = (float*)c_ptr + offset;

                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)uram0);
                    __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)uram1);
                    __tx_fence();

                    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)uram0, (void *)uram1, (void *)uram2);
                    __tx_fence();
                    
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram2, c_elem_ptr);
                    __tx_fence();
                    
                } else {                    
                    // 加载第0块数据
                    int chunk_0_size = (ne0 > MTile) ? MTile : ne0;
                    config_tcsr_vector<aTCSR>(chunk_0_size);
                    config_tcsr_vector<bTCSR>(chunk_0_size);
                    config_tcsr_vector<cTCSR>(chunk_0_size);
                    
                    long offset_0 = i * stride_dim3 + j * stride_dim2 + k * stride_dim1;
                    float* a_elem_ptr_0 = (float*)a_ptr + offset_0;
                    float* b_elem_ptr_0 = (float*)b_ptr + offset_0;
                    
                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_0, (void *)(uram0 + 0 * MTile * sizeof(float)));
                    __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_0, (void *)(uram1 + 0 * MTile * sizeof(float)));
                    
                    // 计算第0块，同时加载第1块
                    if (total_chunks > 1) {
                        int chunk_1_size = ((ne0 - MTile) > MTile) ? MTile : (ne0 - MTile);
                        
                        __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)(uram0 + 0 * MTile * sizeof(float)), 
                                                                      (void *)(uram1 + 0 * MTile * sizeof(float)), 
                                                                      (void *)(uram2 + 0 * MTile * sizeof(float)));

                        config_tcsr_vector<aTCSR>(chunk_1_size);
                        config_tcsr_vector<bTCSR>(chunk_1_size);
                        
                        long offset_1 = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + MTile;
                        float* a_elem_ptr_1 = (float*)a_ptr + offset_1;
                        float* b_elem_ptr_1 = (float*)b_ptr + offset_1;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_1, (void *)(uram0 + 1 * MTile * sizeof(float)));
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_1, (void *)(uram1 + 1 * MTile * sizeof(float)));
                    }
                    
                    for (int chunk_idx = 2; chunk_idx < total_chunks; chunk_idx++) {
                        __tx_fence();
                        
                        long c_offset_prev = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + (chunk_idx - 2) * MTile;
                        float* c_elem_ptr_prev = (float*)c_ptr + c_offset_prev;
                        __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + ((chunk_idx - 2) % 2) * MTile * sizeof(float)), c_elem_ptr_prev);
                        
                        int prev_buffer_idx = (chunk_idx - 1) % 2;
                        __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)(uram0 + prev_buffer_idx * MTile * sizeof(float)), 
                                                                      (void *)(uram1 + prev_buffer_idx * MTile * sizeof(float)), 
                                                                      (void *)(uram2 + prev_buffer_idx * MTile * sizeof(float)));
                        
                        int current_chunk_size = ((ne0 - chunk_idx * MTile) > MTile) ? MTile : (ne0 - chunk_idx * MTile);
                        int current_buffer_idx = chunk_idx % 2;
                        
                        config_tcsr_vector<aTCSR>(current_chunk_size);
                        config_tcsr_vector<bTCSR>(current_chunk_size);
                        
                        long offset_curr = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + chunk_idx * MTile;
                        float* a_elem_ptr_curr = (float*)a_ptr + offset_curr;
                        float* b_elem_ptr_curr = (float*)b_ptr + offset_curr;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_curr, (void *)(uram0 + current_buffer_idx * MTile * sizeof(float)));
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_curr, (void *)(uram1 + current_buffer_idx * MTile * sizeof(float)));
                    }
                    
                    __tx_fence();
                    
                    long c_offset_second_last = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + (total_chunks - 2) * MTile;
                    float* c_elem_ptr_second_last = (float*)c_ptr + c_offset_second_last;
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + 0 * MTile * sizeof(float)), c_elem_ptr_second_last);
                    
                    int last_buffer_idx = 1;
                    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)(uram0 + last_buffer_idx * MTile * sizeof(float)), 
                                                                  (void *)(uram1 + last_buffer_idx * MTile * sizeof(float)), 
                                                                  (void *)(uram2 + last_buffer_idx * MTile * sizeof(float)));
                    
                    long c_offset_last = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + (total_chunks - 1) * MTile;
                    float* c_elem_ptr_last = (float*)c_ptr + c_offset_last;
                    __tx_fence();
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + last_buffer_idx * MTile * sizeof(float)), c_elem_ptr_last);
                    __tx_fence();
                }
            }
        }
    }

    return 0;
}


__device__ long kernel_add_fp32_with_broadcast(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{
    // 计算广播参数
    uint8_t broadcast = get_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                     psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);
        
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

                int total_chunks = (psrc0_ne0 + MTile - 1) / MTile;
                
                if (total_chunks == 1) {
                    // 只有一个chunk，直接处理，无需软流水
                    int current_chunk = psrc0_ne0;
                    
                    config_tcsr_vector<aTCSR>(current_chunk);
                    config_tcsr_vector<bTCSR>(current_chunk);
                    config_tcsr_vector<cTCSR>(current_chunk);

                    long a_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1;
                    float* a_elem_ptr = (float*)a_ptr + a_offset;
                    long b_offset;
                    if (psrc1_ne0 == 1) {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    } else {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    }
                    float* b_elem_ptr = (float*)b_ptr + b_offset;
                    long c_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1;
                    float* c_elem_ptr = (float*)c_ptr + c_offset;

                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)uram0);
                    if (psrc1_ne0 == 1) {
                        config_tcsr_vector<bTCSR>(1);
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)uram1);
                    } else {
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)uram1);
                    }
                    __tx_fence();

                    uint8_t current_broadcast = (psrc1_ne0 == 1) ? (broadcast | 0b0001) : broadcast;
                        
#define CALL_TX_FP_ADD_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)uram0, (void *)uram1, (void *)uram2); \
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
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram2, c_elem_ptr);
                    __tx_fence();
                    
                } else {
                    int chunk_0_size = (psrc0_ne0 > MTile) ? MTile : psrc0_ne0;
                    config_tcsr_vector<aTCSR>(chunk_0_size);
                    config_tcsr_vector<bTCSR>(chunk_0_size);
                    config_tcsr_vector<cTCSR>(chunk_0_size);
                    
                    long a_offset_0 = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1;
                    float* a_elem_ptr_0 = (float*)a_ptr + a_offset_0;
                    long b_offset_0;
                    if (psrc1_ne0 == 1) {
                        b_offset_0 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    } else {
                        b_offset_0 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    }
                    float* b_elem_ptr_0 = (float*)b_ptr + b_offset_0;
                    
                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_0, (void *)(uram0 + 0 * MTile * sizeof(float)));
                    if (psrc1_ne0 == 1) {
                        config_tcsr_vector<bTCSR>(1);
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_0, (void *)(uram1 + 0 * MTile * sizeof(float)));
                    } else {
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_0, (void *)(uram1 + 0 * MTile * sizeof(float)));
                    }
                    
                    uint8_t current_broadcast = (psrc1_ne0 == 1) ? (broadcast | 0b0001) : broadcast;
                    
                    // 计算第0块，同时加载第1块
                    if (total_chunks > 1) {
                        int chunk_1_size = ((psrc0_ne0 - MTile) > MTile) ? MTile : (psrc0_ne0 - MTile);
                        
#define CALL_TX_FP_ADD_FP32_PIPELINE(broadcast_val)                            \
  case broadcast_val:                                                          \
    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)(uram0 + 0 * MTile * sizeof(float)), (void *)(uram1 + 0 * MTile * sizeof(float)), (void *)(uram2 + 0 * MTile * sizeof(float))); \
    break;

                        switch (current_broadcast) {
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0000)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0001)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0010)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0011)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0100)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0101)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0110)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0111)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1000)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1001)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1010)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1011)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1100)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1101)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1110)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1111)
                        default:
                            return -1;
                        }

#undef CALL_TX_FP_ADD_FP32_PIPELINE

                        config_tcsr_vector<aTCSR>(chunk_1_size);
                        config_tcsr_vector<bTCSR>(chunk_1_size);
                        
                        long a_offset_1 = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + MTile;
                        float* a_elem_ptr_1 = (float*)a_ptr + a_offset_1;
                        long b_offset_1;
                        if (psrc1_ne0 == 1) {
                            b_offset_1 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                        } else {
                            b_offset_1 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + MTile;
                        }
                        float* b_elem_ptr_1 = (float*)b_ptr + b_offset_1;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_1, (void *)(uram0 + 1 * MTile * sizeof(float)));
                        if (psrc1_ne0 == 1) {
                            config_tcsr_vector<bTCSR>(1);
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_1, (void *)(uram1 + 1 * MTile * sizeof(float)));
                        } else {
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_1, (void *)(uram1 + 1 * MTile * sizeof(float)));
                        }
                    }
                    
                    for (int chunk_idx = 2; chunk_idx < total_chunks; chunk_idx++) {
                        __tx_fence();
                        
                        long c_offset_prev = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + (chunk_idx - 2) * MTile;
                        float* c_elem_ptr_prev = (float*)c_ptr + c_offset_prev;
                        __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + ((chunk_idx - 2) % 2) * MTile * sizeof(float)), c_elem_ptr_prev);
                        
                        int prev_buffer_idx = (chunk_idx - 1) % 2;
                        
#define CALL_TX_FP_ADD_FP32_MAIN(broadcast_val)                                \
  case broadcast_val:                                                          \
    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)(uram0 + prev_buffer_idx * MTile * sizeof(float)), (void *)(uram1 + prev_buffer_idx * MTile * sizeof(float)), (void *)(uram2 + prev_buffer_idx * MTile * sizeof(float))); \
    break;

                        switch (current_broadcast) {
                            CALL_TX_FP_ADD_FP32_MAIN(0b0000)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0001)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0010)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0011)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0100)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0101)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0110)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0111)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1000)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1001)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1010)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1011)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1100)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1101)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1110)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1111)
                        default:
                            return -1;
                        }

#undef CALL_TX_FP_ADD_FP32_MAIN
                        
                        int current_chunk_size = ((psrc0_ne0 - chunk_idx * MTile) > MTile) ? MTile : (psrc0_ne0 - chunk_idx * MTile);
                        int current_buffer_idx = chunk_idx % 2;
                        
                        config_tcsr_vector<aTCSR>(current_chunk_size);
                        config_tcsr_vector<bTCSR>(current_chunk_size);
                        
                        long a_offset_curr = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + chunk_idx * MTile;
                        float* a_elem_ptr_curr = (float*)a_ptr + a_offset_curr;
                        long b_offset_curr;
                        if (psrc1_ne0 == 1) {
                            b_offset_curr = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                        } else {
                            b_offset_curr = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + chunk_idx * MTile;
                        }
                        float* b_elem_ptr_curr = (float*)b_ptr + b_offset_curr;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_curr, (void *)(uram0 + current_buffer_idx * MTile * sizeof(float)));
                        if (psrc1_ne0 == 1) {
                            config_tcsr_vector<bTCSR>(1);
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_curr, (void *)(uram1 + current_buffer_idx * MTile * sizeof(float)));
                        } else {
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_curr, (void *)(uram1 + current_buffer_idx * MTile * sizeof(float)));
                        }
                    }
                    
                    __tx_fence();

                    long c_offset_second_last = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + (total_chunks - 2) * MTile;
                    float* c_elem_ptr_second_last = (float*)c_ptr + c_offset_second_last;
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + 0 * MTile * sizeof(float)), c_elem_ptr_second_last);

                    int last_buffer_idx = 1;
                    
#define CALL_TX_FP_ADD_FP32_LAST(broadcast_val)                                \
  case broadcast_val:                                                          \
    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)(uram0 + last_buffer_idx * MTile * sizeof(float)), (void *)(uram1 + last_buffer_idx * MTile * sizeof(float)), (void *)(uram2 + last_buffer_idx * MTile * sizeof(float))); \
    break;

                    switch (current_broadcast) {
                        CALL_TX_FP_ADD_FP32_LAST(0b0000)
                        CALL_TX_FP_ADD_FP32_LAST(0b0001)
                        CALL_TX_FP_ADD_FP32_LAST(0b0010)
                        CALL_TX_FP_ADD_FP32_LAST(0b0011)
                        CALL_TX_FP_ADD_FP32_LAST(0b0100)
                        CALL_TX_FP_ADD_FP32_LAST(0b0101)
                        CALL_TX_FP_ADD_FP32_LAST(0b0110)
                        CALL_TX_FP_ADD_FP32_LAST(0b0111)
                        CALL_TX_FP_ADD_FP32_LAST(0b1000)
                        CALL_TX_FP_ADD_FP32_LAST(0b1001)
                        CALL_TX_FP_ADD_FP32_LAST(0b1010)
                        CALL_TX_FP_ADD_FP32_LAST(0b1011)
                        CALL_TX_FP_ADD_FP32_LAST(0b1100)
                        CALL_TX_FP_ADD_FP32_LAST(0b1101)
                        CALL_TX_FP_ADD_FP32_LAST(0b1110)
                        CALL_TX_FP_ADD_FP32_LAST(0b1111)
                    default:
                        return -1;
                    }

#undef CALL_TX_FP_ADD_FP32_LAST

                    long c_offset_last = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + (total_chunks - 1) * MTile;
                    float* c_elem_ptr_last = (float*)c_ptr + c_offset_last;
                    __tx_fence();
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + last_buffer_idx * MTile * sizeof(float)), c_elem_ptr_last);
                    __tx_fence();
                }
            }
        }
    }

    return 0;
}

__device__ long kernel_add_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3) {
    
    if (!needs_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                        psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3)) {
        return kernel_add_fp32_without_broadcast(a_ptr, b_ptr, c_ptr, 
                                           psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3);
    } else {
        return kernel_add_fp32_with_broadcast(a_ptr, b_ptr, c_ptr,
                                             psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                             psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);
    }
}

__device__ long kernel_mul_fp32_without_broadcast(void *a_ptr, void *b_ptr, void *c_ptr,
                                             int ne0, int ne1, int ne2, int ne3) {
    long total_elements = (long)ne3 * ne2 * ne1 * ne0;
    
    int stride_dim1 = ne0;
    int stride_dim2 = ne0 * ne1;
    int stride_dim3 = ne0 * ne1 * ne2;

    for (int i = 0; i < ne3; ++i) {
        for (int j = 0; j < ne2; ++j) {
            for (int k = 0; k < ne1; ++k) {
                int total_chunks = (ne0 + MTile - 1) / MTile;
                
                if (total_chunks == 1) {
                    int current_chunk = ne0;
                    
                    config_tcsr_vector<aTCSR>(current_chunk);
                    config_tcsr_vector<bTCSR>(current_chunk);
                    config_tcsr_vector<cTCSR>(current_chunk);

                    long offset = i * stride_dim3 + j * stride_dim2 + k * stride_dim1;
                    float* a_elem_ptr = (float*)a_ptr + offset;
                    float* b_elem_ptr = (float*)b_ptr + offset;
                    float* c_elem_ptr = (float*)c_ptr + offset;

                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)uram0);
                    __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)uram1);
                    __tx_fence();

                    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)uram0, (void *)uram1, (void *)uram2);
                    __tx_fence();
                    
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram2, c_elem_ptr);
                    __tx_fence();
                    
                } else {
                    int chunk_0_size = (ne0 > MTile) ? MTile : ne0;
                    config_tcsr_vector<aTCSR>(chunk_0_size);
                    config_tcsr_vector<bTCSR>(chunk_0_size);
                    config_tcsr_vector<cTCSR>(chunk_0_size);
                    
                    long offset_0 = i * stride_dim3 + j * stride_dim2 + k * stride_dim1;
                    float* a_elem_ptr_0 = (float*)a_ptr + offset_0;
                    float* b_elem_ptr_0 = (float*)b_ptr + offset_0;
                    
                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_0, (void *)(uram0 + 0 * MTile * sizeof(float)));
                    __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_0, (void *)(uram1 + 0 * MTile * sizeof(float)));
                    
                    // 计算第0块，同时加载第1块
                    if (total_chunks > 1) {
                        int chunk_1_size = ((ne0 - MTile) > MTile) ? MTile : (ne0 - MTile);
                        
                        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)(uram0 + 0 * MTile * sizeof(float)), 
                                                                      (void *)(uram1 + 0 * MTile * sizeof(float)), 
                                                                      (void *)(uram2 + 0 * MTile * sizeof(float)));

                        config_tcsr_vector<aTCSR>(chunk_1_size);
                        config_tcsr_vector<bTCSR>(chunk_1_size);
                        
                        long offset_1 = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + MTile;
                        float* a_elem_ptr_1 = (float*)a_ptr + offset_1;
                        float* b_elem_ptr_1 = (float*)b_ptr + offset_1;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_1, (void *)(uram0 + 1 * MTile * sizeof(float)));
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_1, (void *)(uram1 + 1 * MTile * sizeof(float)));
                    }
                    
                    for (int chunk_idx = 2; chunk_idx < total_chunks; chunk_idx++) {
                        __tx_fence();
                        
                        long c_offset_prev = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + (chunk_idx - 2) * MTile;
                        float* c_elem_ptr_prev = (float*)c_ptr + c_offset_prev;
                        __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + ((chunk_idx - 2) % 2) * MTile * sizeof(float)), c_elem_ptr_prev);
                        
                        int prev_buffer_idx = (chunk_idx - 1) % 2;
                        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)(uram0 + prev_buffer_idx * MTile * sizeof(float)), 
                                                                      (void *)(uram1 + prev_buffer_idx * MTile * sizeof(float)), 
                                                                      (void *)(uram2 + prev_buffer_idx * MTile * sizeof(float)));
                        
                        int current_chunk_size = ((ne0 - chunk_idx * MTile) > MTile) ? MTile : (ne0 - chunk_idx * MTile);
                        int current_buffer_idx = chunk_idx % 2;
                        
                        config_tcsr_vector<aTCSR>(current_chunk_size);
                        config_tcsr_vector<bTCSR>(current_chunk_size);
                        
                        long offset_curr = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + chunk_idx * MTile;
                        float* a_elem_ptr_curr = (float*)a_ptr + offset_curr;
                        float* b_elem_ptr_curr = (float*)b_ptr + offset_curr;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_curr, (void *)(uram0 + current_buffer_idx * MTile * sizeof(float)));
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_curr, (void *)(uram1 + current_buffer_idx * MTile * sizeof(float)));
                    }
                    
                    __tx_fence();
                    
                    long c_offset_second_last = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + (total_chunks - 2) * MTile;
                    float* c_elem_ptr_second_last = (float*)c_ptr + c_offset_second_last;
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + 0 * MTile * sizeof(float)), c_elem_ptr_second_last);
                    
                    int last_buffer_idx = 1;
                    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>((void *)(uram0 + last_buffer_idx * MTile * sizeof(float)), 
                                                                  (void *)(uram1 + last_buffer_idx * MTile * sizeof(float)), 
                                                                  (void *)(uram2 + last_buffer_idx * MTile * sizeof(float)));
                    
                    long c_offset_last = i * stride_dim3 + j * stride_dim2 + k * stride_dim1 + (total_chunks - 1) * MTile;
                    float* c_elem_ptr_last = (float*)c_ptr + c_offset_last;
                    __tx_fence();
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + last_buffer_idx * MTile * sizeof(float)), c_elem_ptr_last);
                    __tx_fence();
                }
            }
        }
    }

    return 0;
}


__device__ long kernel_mul_fp32_with_broadcast(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{
    // 计算广播参数
    uint8_t broadcast = get_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                     psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);
        
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

                // 软流水优化的向量化处理
                int total_chunks = (psrc0_ne0 + MTile - 1) / MTile;
                
                if (total_chunks == 1) {
                    // 只有一个chunk，直接处理，无需软流水
                    int current_chunk = psrc0_ne0;
                    
                    config_tcsr_vector<aTCSR>(current_chunk);
                    config_tcsr_vector<bTCSR>(current_chunk);
                    config_tcsr_vector<cTCSR>(current_chunk);

                    long a_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1;
                    float* a_elem_ptr = (float*)a_ptr + a_offset;
                    long b_offset;
                    if (psrc1_ne0 == 1) {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    } else {
                        b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    }
                    float* b_elem_ptr = (float*)b_ptr + b_offset;
                    long c_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1;
                    float* c_elem_ptr = (float*)c_ptr + c_offset;

                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)uram0);
                    if (psrc1_ne0 == 1) {
                        config_tcsr_vector<bTCSR>(1);
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)uram1);
                    } else {
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)uram1);
                    }
                    __tx_fence();

                    uint8_t current_broadcast = (psrc1_ne0 == 1) ? (broadcast | 0b0001) : broadcast;
                        
#define CALL_TX_FP_ADD_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)uram0, (void *)uram1, (void *)uram2); \
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
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram2, c_elem_ptr);
                    __tx_fence();
                    
                } else {
                    
                    // 预处理：加载第0块数据
                    int chunk_0_size = (psrc0_ne0 > MTile) ? MTile : psrc0_ne0;
                    config_tcsr_vector<aTCSR>(chunk_0_size);
                    config_tcsr_vector<bTCSR>(chunk_0_size);
                    config_tcsr_vector<cTCSR>(chunk_0_size);
                    
                    long a_offset_0 = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1;
                    float* a_elem_ptr_0 = (float*)a_ptr + a_offset_0;
                    long b_offset_0;
                    if (psrc1_ne0 == 1) {
                        b_offset_0 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    } else {
                        b_offset_0 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                    }
                    float* b_elem_ptr_0 = (float*)b_ptr + b_offset_0;
                    
                    __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_0, (void *)(uram0 + 0 * MTile * sizeof(float)));
                    if (psrc1_ne0 == 1) {
                        config_tcsr_vector<bTCSR>(1);
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_0, (void *)(uram1 + 0 * MTile * sizeof(float)));
                    } else {
                        __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_0, (void *)(uram1 + 0 * MTile * sizeof(float)));
                    }
                    
                    uint8_t current_broadcast = (psrc1_ne0 == 1) ? (broadcast | 0b0001) : broadcast;
                    
                    // 计算第0块，同时加载第1块
                    if (total_chunks > 1) {
                        int chunk_1_size = ((psrc0_ne0 - MTile) > MTile) ? MTile : (psrc0_ne0 - MTile);
                        
#define CALL_TX_FP_ADD_FP32_PIPELINE(broadcast_val)                            \
  case broadcast_val:                                                          \
    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)(uram0 + 0 * MTile * sizeof(float)), (void *)(uram1 + 0 * MTile * sizeof(float)), (void *)(uram2 + 0 * MTile * sizeof(float))); \
    break;

                        switch (current_broadcast) {
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0000)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0001)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0010)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0011)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0100)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0101)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0110)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b0111)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1000)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1001)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1010)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1011)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1100)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1101)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1110)
                            CALL_TX_FP_ADD_FP32_PIPELINE(0b1111)
                        default:
                            return -1;
                        }

#undef CALL_TX_FP_ADD_FP32_PIPELINE

                        config_tcsr_vector<aTCSR>(chunk_1_size);
                        config_tcsr_vector<bTCSR>(chunk_1_size);
                        
                        long a_offset_1 = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + MTile;
                        float* a_elem_ptr_1 = (float*)a_ptr + a_offset_1;
                        long b_offset_1;
                        if (psrc1_ne0 == 1) {
                            b_offset_1 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                        } else {
                            b_offset_1 = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + MTile;
                        }
                        float* b_elem_ptr_1 = (float*)b_ptr + b_offset_1;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_1, (void *)(uram0 + 1 * MTile * sizeof(float)));
                        if (psrc1_ne0 == 1) {
                            config_tcsr_vector<bTCSR>(1);
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_1, (void *)(uram1 + 1 * MTile * sizeof(float)));
                        } else {
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_1, (void *)(uram1 + 1 * MTile * sizeof(float)));
                        }
                    }
                    
                    for (int chunk_idx = 2; chunk_idx < total_chunks; chunk_idx++) {
                        __tx_fence();
                        
                        long c_offset_prev = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + (chunk_idx - 2) * MTile;
                        float* c_elem_ptr_prev = (float*)c_ptr + c_offset_prev;
                        __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + ((chunk_idx - 2) % 2) * MTile * sizeof(float)), c_elem_ptr_prev);

                        int prev_buffer_idx = (chunk_idx - 1) % 2;
                        
#define CALL_TX_FP_ADD_FP32_MAIN(broadcast_val)                                \
  case broadcast_val:                                                          \
    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)(uram0 + prev_buffer_idx * MTile * sizeof(float)), (void *)(uram1 + prev_buffer_idx * MTile * sizeof(float)), (void *)(uram2 + prev_buffer_idx * MTile * sizeof(float))); \
    break;

                        switch (current_broadcast) {
                            CALL_TX_FP_ADD_FP32_MAIN(0b0000)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0001)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0010)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0011)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0100)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0101)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0110)
                            CALL_TX_FP_ADD_FP32_MAIN(0b0111)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1000)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1001)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1010)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1011)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1100)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1101)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1110)
                            CALL_TX_FP_ADD_FP32_MAIN(0b1111)
                        default:
                            return -1;
                        }

#undef CALL_TX_FP_ADD_FP32_MAIN

                        int current_chunk_size = ((psrc0_ne0 - chunk_idx * MTile) > MTile) ? MTile : (psrc0_ne0 - chunk_idx * MTile);
                        int current_buffer_idx = chunk_idx % 2;
                        
                        config_tcsr_vector<aTCSR>(current_chunk_size);
                        config_tcsr_vector<bTCSR>(current_chunk_size);
                        
                        long a_offset_curr = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + chunk_idx * MTile;
                        float* a_elem_ptr_curr = (float*)a_ptr + a_offset_curr;
                        long b_offset_curr;
                        if (psrc1_ne0 == 1) {
                            b_offset_curr = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1;
                        } else {
                            b_offset_curr = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + chunk_idx * MTile;
                        }
                        float* b_elem_ptr_curr = (float*)b_ptr + b_offset_curr;
                        
                        __tx_async_uram_load<aTCSR, aTCSR, 2>(a_elem_ptr_curr, (void *)(uram0 + current_buffer_idx * MTile * sizeof(float)));
                        if (psrc1_ne0 == 1) {
                            config_tcsr_vector<bTCSR>(1);
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_curr, (void *)(uram1 + current_buffer_idx * MTile * sizeof(float)));
                        } else {
                            __tx_async_uram_load<bTCSR, bTCSR, 2>(b_elem_ptr_curr, (void *)(uram1 + current_buffer_idx * MTile * sizeof(float)));
                        }
                    }
                    
                    // 最后两个chunk
                    __tx_fence();

                    long c_offset_second_last = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + (total_chunks - 2) * MTile;
                    float* c_elem_ptr_second_last = (float*)c_ptr + c_offset_second_last;
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + 0 * MTile * sizeof(float)), c_elem_ptr_second_last);

                    int last_buffer_idx = 1;
                    
#define CALL_TX_FP_ADD_FP32_LAST(broadcast_val)                                \
  case broadcast_val:                                                          \
    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, broadcast_val>((void *)(uram0 + last_buffer_idx * MTile * sizeof(float)), (void *)(uram1 + last_buffer_idx * MTile * sizeof(float)), (void *)(uram2 + last_buffer_idx * MTile * sizeof(float))); \
    break;

                    switch (current_broadcast) {
                        CALL_TX_FP_ADD_FP32_LAST(0b0000)
                        CALL_TX_FP_ADD_FP32_LAST(0b0001)
                        CALL_TX_FP_ADD_FP32_LAST(0b0010)
                        CALL_TX_FP_ADD_FP32_LAST(0b0011)
                        CALL_TX_FP_ADD_FP32_LAST(0b0100)
                        CALL_TX_FP_ADD_FP32_LAST(0b0101)
                        CALL_TX_FP_ADD_FP32_LAST(0b0110)
                        CALL_TX_FP_ADD_FP32_LAST(0b0111)
                        CALL_TX_FP_ADD_FP32_LAST(0b1000)
                        CALL_TX_FP_ADD_FP32_LAST(0b1001)
                        CALL_TX_FP_ADD_FP32_LAST(0b1010)
                        CALL_TX_FP_ADD_FP32_LAST(0b1011)
                        CALL_TX_FP_ADD_FP32_LAST(0b1100)
                        CALL_TX_FP_ADD_FP32_LAST(0b1101)
                        CALL_TX_FP_ADD_FP32_LAST(0b1110)
                        CALL_TX_FP_ADD_FP32_LAST(0b1111)
                    default:
                        return -1;
                    }

#undef CALL_TX_FP_ADD_FP32_LAST
                    
                    // 存储最后一个chunk的结果
                    long c_offset_last = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + (total_chunks - 1) * MTile;
                    float* c_elem_ptr_last = (float*)c_ptr + c_offset_last;
                    __tx_fence();
                    __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)(uram2 + last_buffer_idx * MTile * sizeof(float)), c_elem_ptr_last);
                    __tx_fence();
                }
            }
        }
    }

    return 0;
}

__device__ long kernel_mul_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3) {
    
    if (!needs_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                        psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3)) {
        return kernel_mul_fp32_without_broadcast(a_ptr, b_ptr, c_ptr, 
                                           psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3);
    } else {
        return kernel_mul_fp32_with_broadcast(a_ptr, b_ptr, c_ptr,
                                             psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                             psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);
    }
}