#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <cmath>
#include <stdio.h>
#include <string.h>

static constexpr long aTCSR = 0;
static constexpr long bTCSR = 1;
static constexpr long cTCSR = 2;

static constexpr long lsuTCSR = 0;

static constexpr int MTile = 1024;

extern int _uram[256 * 1024];

template <int TCSR_ID>
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


template <long GROUPBASE, typename elemty>
static inline void configLSUTCSR_1d(int m) {
    long m_bytes = (long)m * (long)sizeof(elemty);
    m_bytes = m_bytes < 64 ? 64 : m_bytes;

    if ((m_bytes % 64) != 0) {
        // 暂不支持非64B对齐传输
        printf("LSU transfer size must be aligned to 64B, but got m_bytes=%ld\n", m_bytes);
    }

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(((m_bytes + 63) / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(m_bytes);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(m_bytes);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
}

static inline void* uram_ptr(size_t float_offset) {
    return (void*)(((float*)_uram) + float_offset);
}

__device__ long kernel_add_fp32_with_broadcast(void* a_ptr, void* b_ptr, void* c_ptr,
                                               int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                               int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3) {
    
    // 1. 计算维度步长
    const int a_stride_dim1 = psrc0_ne0;
    const int a_stride_dim2 = psrc0_ne0 * psrc0_ne1;
    const int a_stride_dim3 = psrc0_ne0 * psrc0_ne1 * psrc0_ne2;

    const int b_stride_dim1 = psrc1_ne0;
    const int b_stride_dim2 = psrc1_ne0 * psrc1_ne1;
    const int b_stride_dim3 = psrc1_ne0 * psrc1_ne1 * psrc1_ne2;

    // URAM 布局常量
    constexpr size_t A_base = 0;
    constexpr size_t B_base = A_base + 2 * MTile;
    constexpr size_t C_base = B_base + 2 * MTile;

    for (int i = 0; i < psrc0_ne3; ++i) {
        int i1 = (psrc1_ne3 == 1) ? 0 : i;

        for (int j = 0; j < psrc0_ne2; ++j) {
            int j1 = (psrc1_ne2 == 1) ? 0 : j;

            for (int k = 0; k < psrc0_ne1; ++k) {
                int k1 = (psrc1_ne1 == 1) ? 0 : k;

                const int total_chunks = (psrc0_ne0 + MTile - 1) / MTile;
                if (total_chunks <= 0) continue;

                const long a_base_offset = (long)i * a_stride_dim3 + (long)j * a_stride_dim2 + (long)k * a_stride_dim1;
                // B 的偏移计算已经包含了 Dim1/2/3 的广播逻辑 
                const long b_base_offset = (long)i1 * b_stride_dim3 + (long)j1 * b_stride_dim2 + (long)k1 * b_stride_dim1;
                const long c_base_offset = a_base_offset;

                // 外层维度的广播由软件循环处理，指令只关心 Dim0
                uint8_t instruction_mask = (psrc1_ne0 == 1) ? 0b0001 : 0b0000;

                // 预加载阶段
                int chunk0_size = (psrc0_ne0 > MTile) ? MTile : psrc0_ne0;
                
                configLSUTCSR_1d<lsuTCSR, float>(chunk0_size);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)((float*)a_ptr + a_base_offset));

                // B 的加载
                int b_chunk0_size = (psrc1_ne0 == 1) ? 1 : chunk0_size;
                configLSUTCSR_1d<lsuTCSR, float>(b_chunk0_size);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + 0 * MTile), (long)((float*)b_ptr + b_base_offset));

                config_tcsr_vector<aTCSR>(chunk0_size);
                config_tcsr_vector<bTCSR>((psrc1_ne0 == 1) ? 1 : chunk0_size);
                config_tcsr_vector<cTCSR>(chunk0_size);

                __tx_lsu_sb_bar();
                __tx_sb_bar();


                for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                    const int cur_buf = chunk_idx % 2;
                    const int next_buf = (chunk_idx + 1) % 2;
                    const int prev_buf = (chunk_idx - 1 + 2) % 2;

                    // 预取
                    if (chunk_idx < total_chunks - 1) {
                        int next_chunk_size = (psrc0_ne0 - (chunk_idx + 1) * MTile > MTile) ? MTile : (psrc0_ne0 - (chunk_idx + 1) * MTile);
                        
                        configLSUTCSR_1d<lsuTCSR, float>(next_chunk_size);
                        long a_next_off = a_base_offset + (long)(chunk_idx + 1) * MTile;
                        tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)((float*)a_ptr + a_next_off));

                        if (psrc1_ne0 == 1) {
                            configLSUTCSR_1d<lsuTCSR, float>(1);
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)((float*)b_ptr + b_base_offset));
                        } else {
                            configLSUTCSR_1d<lsuTCSR, float>(next_chunk_size);
                            long b_next_off = b_base_offset + (long)(chunk_idx + 1) * MTile;
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)((float*)b_ptr + b_next_off));
                        }
                    }

                    // 回写
                    if (chunk_idx > 0) {
                        int prev_chunk_size = (psrc0_ne0 - (chunk_idx - 1) * MTile > MTile) ? MTile : (psrc0_ne0 - (chunk_idx - 1) * MTile);
                        configLSUTCSR_1d<lsuTCSR, float>(prev_chunk_size);
                        
                        long c_prev_off = c_base_offset + (long)(chunk_idx - 1) * MTile;
                        tx_lsu_dma_u2d<lsuTCSR>((long)((float*)c_ptr + c_prev_off), (long)uram_ptr(C_base + prev_buf * MTile));
                    }

                    // 计算 Cur
                    int cur_size = (psrc0_ne0 - chunk_idx * MTile > MTile) ? MTile : (psrc0_ne0 - chunk_idx * MTile);
                    
                    config_tcsr_vector<aTCSR>(cur_size);
                    config_tcsr_vector<bTCSR>((psrc1_ne0 == 1) ? 1 : cur_size);
                    config_tcsr_vector<cTCSR>(cur_size);

                    void* a_uram = uram_ptr(A_base + cur_buf * MTile);
                    void* b_uram = uram_ptr(B_base + cur_buf * MTile);
                    void* c_uram = uram_ptr(C_base + cur_buf * MTile);

                    if (instruction_mask == 0b0001) {
                        __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0001>(a_uram, b_uram, c_uram);
                    } else {
                        __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(a_uram, b_uram, c_uram);
                    }

                    __tx_sb_bar();
                    __tx_lsu_sb_bar();
                }

                // 回写
                int last_idx = total_chunks - 1;
                int last_buf = last_idx % 2;
                int last_size = (psrc0_ne0 - last_idx * MTile > MTile) ? MTile : (psrc0_ne0 - last_idx * MTile);

                configLSUTCSR_1d<lsuTCSR, float>(last_size);
                long c_last_off = c_base_offset + (long)last_idx * MTile;
                tx_lsu_dma_u2d<lsuTCSR>((long)((float*)c_ptr + c_last_off), (long)uram_ptr(C_base + last_buf * MTile));
                __tx_lsu_sb_bar();
            }
        }
    }
    return 0;
}

__device__ long kernel_mul_fp32_with_broadcast(void* a_ptr, void* b_ptr, void* c_ptr,
                                               int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                               int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3) {
    
    const int a_stride_dim1 = psrc0_ne0;
    const int a_stride_dim2 = psrc0_ne0 * psrc0_ne1;
    const int a_stride_dim3 = psrc0_ne0 * psrc0_ne1 * psrc0_ne2;

    const int b_stride_dim1 = psrc1_ne0;
    const int b_stride_dim2 = psrc1_ne0 * psrc1_ne1;
    const int b_stride_dim3 = psrc1_ne0 * psrc1_ne1 * psrc1_ne2;

    constexpr size_t A_base = 0;
    constexpr size_t B_base = A_base + 2 * MTile;
    constexpr size_t C_base = B_base + 2 * MTile;

    for (int i = 0; i < psrc0_ne3; ++i) {
        int i1 = (psrc1_ne3 == 1) ? 0 : i;
        for (int j = 0; j < psrc0_ne2; ++j) {
            int j1 = (psrc1_ne2 == 1) ? 0 : j;
            for (int k = 0; k < psrc0_ne1; ++k) {
                int k1 = (psrc1_ne1 == 1) ? 0 : k;

                const int total_chunks = (psrc0_ne0 + MTile - 1) / MTile;
                if (total_chunks <= 0) continue;

                const long a_base_offset = (long)i * a_stride_dim3 + (long)j * a_stride_dim2 + (long)k * a_stride_dim1;
                const long b_base_offset = (long)i1 * b_stride_dim3 + (long)j1 * b_stride_dim2 + (long)k1 * b_stride_dim1;
                const long c_base_offset = a_base_offset; 

                uint8_t instruction_mask = (psrc1_ne0 == 1) ? 0b0001 : 0b0000;

                int chunk0_size = (psrc0_ne0 > MTile) ? MTile : psrc0_ne0;
                configLSUTCSR_1d<lsuTCSR, float>(chunk0_size);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)((float*)a_ptr + a_base_offset));

                int b_chunk0_size = (psrc1_ne0 == 1) ? 1 : chunk0_size;
                configLSUTCSR_1d<lsuTCSR, float>(b_chunk0_size);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + 0 * MTile), (long)((float*)b_ptr + b_base_offset));

                config_tcsr_vector<aTCSR>(chunk0_size);
                config_tcsr_vector<bTCSR>((psrc1_ne0 == 1) ? 1 : chunk0_size);
                config_tcsr_vector<cTCSR>(chunk0_size);

                __tx_lsu_sb_bar();
                __tx_sb_bar();

                for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                    const int cur_buf = chunk_idx % 2;
                    const int next_buf = (chunk_idx + 1) % 2;
                    const int prev_buf = (chunk_idx - 1 + 2) % 2;

                    if (chunk_idx < total_chunks - 1) {
                        int next_chunk_size = (psrc0_ne0 - (chunk_idx + 1) * MTile > MTile) ? MTile : (psrc0_ne0 - (chunk_idx + 1) * MTile);
                        
                        configLSUTCSR_1d<lsuTCSR, float>(next_chunk_size);
                        long a_next_off = a_base_offset + (long)(chunk_idx + 1) * MTile;
                        tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)((float*)a_ptr + a_next_off));

                        if (psrc1_ne0 == 1) {
                            configLSUTCSR_1d<lsuTCSR, float>(1);
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)((float*)b_ptr + b_base_offset));
                        } else {
                            configLSUTCSR_1d<lsuTCSR, float>(next_chunk_size);
                            long b_next_off = b_base_offset + (long)(chunk_idx + 1) * MTile;
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)((float*)b_ptr + b_next_off));
                        }
                    }

                    if (chunk_idx > 0) {
                        int prev_chunk_size = (psrc0_ne0 - (chunk_idx - 1) * MTile > MTile) ? MTile : (psrc0_ne0 - (chunk_idx - 1) * MTile);
                        configLSUTCSR_1d<lsuTCSR, float>(prev_chunk_size);
                        long c_prev_off = c_base_offset + (long)(chunk_idx - 1) * MTile;
                        tx_lsu_dma_u2d<lsuTCSR>((long)((float*)c_ptr + c_prev_off), (long)uram_ptr(C_base + prev_buf * MTile));
                    }

                    int cur_size = (psrc0_ne0 - chunk_idx * MTile > MTile) ? MTile : (psrc0_ne0 - chunk_idx * MTile);
                    config_tcsr_vector<aTCSR>(cur_size);
                    config_tcsr_vector<bTCSR>((psrc1_ne0 == 1) ? 1 : cur_size);
                    config_tcsr_vector<cTCSR>(cur_size);

                    void* a_uram = uram_ptr(A_base + cur_buf * MTile);
                    void* b_uram = uram_ptr(B_base + cur_buf * MTile);
                    void* c_uram = uram_ptr(C_base + cur_buf * MTile);

                    if (instruction_mask == 0b0001) {
                        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0001>(a_uram, b_uram, c_uram);
                    } else {
                        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(a_uram, b_uram, c_uram);
                    }

                    __tx_sb_bar();
                    __tx_lsu_sb_bar();
                }

                int last_idx = total_chunks - 1;
                int last_buf = last_idx % 2;
                int last_size = (psrc0_ne0 - last_idx * MTile > MTile) ? MTile : (psrc0_ne0 - last_idx * MTile);

                configLSUTCSR_1d<lsuTCSR, float>(last_size);
                long c_last_off = c_base_offset + (long)last_idx * MTile;
                tx_lsu_dma_u2d<lsuTCSR>((long)((float*)c_ptr + c_last_off), (long)uram_ptr(C_base + last_buf * MTile));
                __tx_lsu_sb_bar();
            }
        }
    }
    return 0;
}

__device__ long kernel_add_fp32(void* a_ptr, void* b_ptr, void* c_ptr,
                                  int a0,int a1,int a2,int a3,
                                  int b0,int b1,int b2,int b3) {
    return kernel_add_fp32_with_broadcast(a_ptr,b_ptr,c_ptr, a0,a1,a2,a3, b0,b1,b2,b3);
}

__device__ long kernel_mul_fp32(void* a_ptr, void* b_ptr, void* c_ptr,
                                  int a0,int a1,int a2,int a3,
                                  int b0,int b1,int b2,int b3) {
    return kernel_mul_fp32_with_broadcast(a_ptr,b_ptr,c_ptr, a0,a1,a2,a3, b0,b1,b2,b3);
}


