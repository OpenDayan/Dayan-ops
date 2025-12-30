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

__device__ long kernel_add_fp32_without_broadcast(void* a_ptr,
                                                    void* b_ptr,
                                                    void* c_ptr,
                                                    int ne0, int ne1, int ne2, int ne3) {
    const int stride_dim1 = ne0;
    const int stride_dim2 = ne0 * ne1;
    const int stride_dim3 = ne0 * ne1 * ne2;

    constexpr size_t A_base = 0;
    constexpr size_t B_base = A_base + 2 * MTile;
    constexpr size_t C_base = B_base + 2 * MTile;

    for (int i = 0; i < ne3; ++i) {
        for (int j = 0; j < ne2; ++j) {
            for (int k = 0; k < ne1; ++k) {
                const int total_chunks = (ne0 + MTile - 1) / MTile;
                if (total_chunks <= 0) continue;

                const long base_offset =
                    (long)i * stride_dim3 + (long)j * stride_dim2 + (long)k * stride_dim1;

                // 预加载 chunk0
                int chunk0 = (ne0 > MTile) ? MTile : ne0;
                config_tcsr_vector<aTCSR>(chunk0);
                config_tcsr_vector<bTCSR>(chunk0);
                config_tcsr_vector<cTCSR>(chunk0);
                configLSUTCSR_1d<lsuTCSR, float>(chunk0);

                float* a0 = (float*)a_ptr + base_offset;
                float* b0 = (float*)b_ptr + base_offset;

                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)a0);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + 0 * MTile), (long)b0);
                __tx_lsu_sb_bar();
                __tx_sb_bar();

                // 主循环
                for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                    const int cur_buf = chunk_idx % 2;
                    const int next_buf = (chunk_idx + 1) % 2;
                    const int prev_buf = (chunk_idx - 1 + 2) % 2;

                    // 预取 next
                    if (chunk_idx < total_chunks - 1) {
                        const int next_size =
                            (ne0 - (chunk_idx + 1) * MTile > MTile)
                                ? MTile
                                : (ne0 - (chunk_idx + 1) * MTile);

                        configLSUTCSR_1d<lsuTCSR, float>(next_size);

                        long next_offset = base_offset + (long)(chunk_idx + 1) * MTile;
                        float* a_next = (float*)a_ptr + next_offset;
                        float* b_next = (float*)b_ptr + next_offset;

                        tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)a_next);
                        tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)b_next);
                    }

                    // 回写 prev
                    if (chunk_idx > 0) {
                        const int prev_size =
                            (ne0 - (chunk_idx - 1) * MTile > MTile)
                                ? MTile
                                : (ne0 - (chunk_idx - 1) * MTile);

                        configLSUTCSR_1d<lsuTCSR, float>(prev_size);

                        long prev_offset = base_offset + (long)(chunk_idx - 1) * MTile;
                        float* c_prev = (float*)c_ptr + prev_offset;

                        tx_lsu_dma_u2d<lsuTCSR>((long)c_prev, (long)uram_ptr(C_base + prev_buf * MTile));
                    }

                    // 计算 cur
                    const int cur_size =
                        (ne0 - chunk_idx * MTile > MTile)
                            ? MTile
                            : (ne0 - chunk_idx * MTile);

                    config_tcsr_vector<aTCSR>(cur_size);
                    config_tcsr_vector<bTCSR>(cur_size);
                    config_tcsr_vector<cTCSR>(cur_size);

                    __tx_fp_add_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(
                        uram_ptr(A_base + cur_buf * MTile),
                        uram_ptr(B_base + cur_buf * MTile),
                        uram_ptr(C_base + cur_buf * MTile));

                    __tx_sb_bar();
                    __tx_lsu_sb_bar();
                }

                // 回写最后一个 chunk
                const int last_idx = total_chunks - 1;
                const int last_buf = last_idx % 2;
                const int last_size =
                    (ne0 - last_idx * MTile > MTile) ? MTile : (ne0 - last_idx * MTile);

                configLSUTCSR_1d<lsuTCSR, float>(last_size);

                long last_offset = base_offset + (long)last_idx * MTile;
                float* c_last = (float*)c_ptr + last_offset;

                tx_lsu_dma_u2d<lsuTCSR>((long)c_last, (long)uram_ptr(C_base + last_buf * MTile));
                __tx_lsu_sb_bar();
            }
        }
    }
    return 0;
}

__device__ long kernel_mul_fp32_without_broadcast(void* a_ptr,
                                                    void* b_ptr,
                                                    void* c_ptr,
                                                    int ne0, int ne1, int ne2, int ne3) {
    const int stride_dim1 = ne0;
    const int stride_dim2 = ne0 * ne1;
    const int stride_dim3 = ne0 * ne1 * ne2;

    constexpr size_t A_base = 0;
    constexpr size_t B_base = A_base + 2 * MTile;
    constexpr size_t C_base = B_base + 2 * MTile;

    for (int i = 0; i < ne3; ++i) {
        for (int j = 0; j < ne2; ++j) {
            for (int k = 0; k < ne1; ++k) {
                const int total_chunks = (ne0 + MTile - 1) / MTile;
                if (total_chunks <= 0) continue;

                const long base_offset =
                    (long)i * stride_dim3 + (long)j * stride_dim2 + (long)k * stride_dim1;

                int chunk0 = (ne0 > MTile) ? MTile : ne0;
                config_tcsr_vector<aTCSR>(chunk0);
                config_tcsr_vector<bTCSR>(chunk0);
                config_tcsr_vector<cTCSR>(chunk0);
                configLSUTCSR_1d<lsuTCSR, float>(chunk0);

                float* a0 = (float*)a_ptr + base_offset;
                float* b0 = (float*)b_ptr + base_offset;

                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)a0);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + 0 * MTile), (long)b0);
                __tx_lsu_sb_bar();
                __tx_sb_bar();

                for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                    const int cur_buf = chunk_idx % 2;
                    const int next_buf = (chunk_idx + 1) % 2;
                    const int prev_buf = (chunk_idx - 1 + 2) % 2;

                    if (chunk_idx < total_chunks - 1) {
                        const int next_size =
                            (ne0 - (chunk_idx + 1) * MTile > MTile)
                                ? MTile
                                : (ne0 - (chunk_idx + 1) * MTile);

                        configLSUTCSR_1d<lsuTCSR, float>(next_size);

                        long next_offset = base_offset + (long)(chunk_idx + 1) * MTile;
                        float* a_next = (float*)a_ptr + next_offset;
                        float* b_next = (float*)b_ptr + next_offset;

                        tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)a_next);
                        tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)b_next);
                    }

                    if (chunk_idx > 0) {
                        const int prev_size =
                            (ne0 - (chunk_idx - 1) * MTile > MTile)
                                ? MTile
                                : (ne0 - (chunk_idx - 1) * MTile);

                        configLSUTCSR_1d<lsuTCSR, float>(prev_size);

                        long prev_offset = base_offset + (long)(chunk_idx - 1) * MTile;
                        float* c_prev = (float*)c_ptr + prev_offset;

                        tx_lsu_dma_u2d<lsuTCSR>((long)c_prev, (long)uram_ptr(C_base + prev_buf * MTile));
                    }

                    const int cur_size =
                        (ne0 - chunk_idx * MTile > MTile)
                            ? MTile
                            : (ne0 - chunk_idx * MTile);

                    config_tcsr_vector<aTCSR>(cur_size);
                    config_tcsr_vector<bTCSR>(cur_size);
                    config_tcsr_vector<cTCSR>(cur_size);

                    __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(
                        uram_ptr(A_base + cur_buf * MTile),
                        uram_ptr(B_base + cur_buf * MTile),
                        uram_ptr(C_base + cur_buf * MTile));

                    __tx_sb_bar();
                    __tx_lsu_sb_bar();
                }

                const int last_idx = total_chunks - 1;
                const int last_buf = last_idx % 2;
                const int last_size =
                    (ne0 - last_idx * MTile > MTile) ? MTile : (ne0 - last_idx * MTile);

                configLSUTCSR_1d<lsuTCSR, float>(last_size);

                long last_offset = base_offset + (long)last_idx * MTile;
                float* c_last = (float*)c_ptr + last_offset;

                tx_lsu_dma_u2d<lsuTCSR>((long)c_last, (long)uram_ptr(C_base + last_buf * MTile));
                __tx_lsu_sb_bar();
            }
        }
    }
    return 0;
}

static inline bool needs_broadcast(int a0, int a1, int a2, int a3,
                                   int b0, int b1, int b2, int b3) {
    return (a0 != b0 || a1 != b1 || a2 != b2 || a3 != b3);
}

#define DISPATCH_VEC_ADD_FP32(mask, a_u, b_u, c_u)                     \
    do {                                                               \
        switch (mask) {                                                \
        case 0b0000: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0000>(a_u,b_u,c_u); break; \
        case 0b0001: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0001>(a_u,b_u,c_u); break; \
        case 0b0010: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0010>(a_u,b_u,c_u); break; \
        case 0b0011: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0011>(a_u,b_u,c_u); break; \
        case 0b0100: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0100>(a_u,b_u,c_u); break; \
        case 0b0101: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0101>(a_u,b_u,c_u); break; \
        case 0b0110: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0110>(a_u,b_u,c_u); break; \
        case 0b0111: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0111>(a_u,b_u,c_u); break; \
        case 0b1000: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1000>(a_u,b_u,c_u); break; \
        case 0b1001: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1001>(a_u,b_u,c_u); break; \
        case 0b1010: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1010>(a_u,b_u,c_u); break; \
        case 0b1011: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1011>(a_u,b_u,c_u); break; \
        case 0b1100: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1100>(a_u,b_u,c_u); break; \
        case 0b1101: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1101>(a_u,b_u,c_u); break; \
        case 0b1110: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1110>(a_u,b_u,c_u); break; \
        case 0b1111: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b1111>(a_u,b_u,c_u); break; \
        default: __tx_fp_add_fp32<aTCSR,bTCSR,cTCSR,0b0000>(a_u,b_u,c_u); break;     \
        }                                                              \
    } while (0)

#define DISPATCH_VEC_MUL_FP32(mask, a_u, b_u, c_u)                     \
    do {                                                               \
        switch (mask) {                                                \
        case 0b0000: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0000>(a_u,b_u,c_u); break; \
        case 0b0001: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0001>(a_u,b_u,c_u); break; \
        case 0b0010: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0010>(a_u,b_u,c_u); break; \
        case 0b0011: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0011>(a_u,b_u,c_u); break; \
        case 0b0100: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0100>(a_u,b_u,c_u); break; \
        case 0b0101: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0101>(a_u,b_u,c_u); break; \
        case 0b0110: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0110>(a_u,b_u,c_u); break; \
        case 0b0111: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0111>(a_u,b_u,c_u); break; \
        case 0b1000: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1000>(a_u,b_u,c_u); break; \
        case 0b1001: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1001>(a_u,b_u,c_u); break; \
        case 0b1010: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1010>(a_u,b_u,c_u); break; \
        case 0b1011: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1011>(a_u,b_u,c_u); break; \
        case 0b1100: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1100>(a_u,b_u,c_u); break; \
        case 0b1101: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1101>(a_u,b_u,c_u); break; \
        case 0b1110: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1110>(a_u,b_u,c_u); break; \
        case 0b1111: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b1111>(a_u,b_u,c_u); break; \
        default: __tx_fp_mul_fp32<aTCSR,bTCSR,cTCSR,0b0000>(a_u,b_u,c_u); break;     \
        }                                                              \
    } while (0)

static inline uint8_t get_broadcast(int a_dim0, int a_dim1, int a_dim2, int a_dim3, int b_dim0,
                                    int b_dim1, int b_dim2, int b_dim3) {
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
    
__device__ long kernel_add_fp32_with_broadcast(void* a_ptr, void* b_ptr, void* c_ptr,
                                                 int a0,int a1,int a2,int a3,
                                                 int b0,int b1,int b2,int b3) {
    const uint8_t broadcast_base = get_broadcast(a0,a1,a2,a3, b0,b1,b2,b3);

    const int a_s1 = a0;
    const int a_s2 = a0 * a1;
    const int a_s3 = a0 * a1 * a2;

    const int b_s1 = b0;
    const int b_s2 = b0 * b1;
    const int b_s3 = b0 * b1 * b2;

    constexpr size_t A_base = 0;
    constexpr size_t B_base = A_base + 2 * MTile;
    constexpr size_t C_base = B_base + 2 * MTile;
    constexpr size_t B_scalar_base = C_base + 2 * MTile;
    (void)B_scalar_base;

    for (int i = 0; i < a3; ++i) {
        const int i1 = (b3 == 1) ? 0 : i;
        for (int j = 0; j < a2; ++j) {
            const int j1 = (b2 == 1) ? 0 : j;
            for (int k = 0; k < a1; ++k) {
                const int k1 = (b1 == 1) ? 0 : k;

                const int total_chunks = (a0 + MTile - 1) / MTile;
                if (total_chunks <= 0) continue;

                const long a_base = (long)i * a_s3 + (long)j * a_s2 + (long)k * a_s1;
                const long c_base = a_base;

                const long b_base =
                    (long)i1 * b_s3 + (long)j1 * b_s2 + (long)k1 * b_s1;

                if (b0 == 1) {
                    configLSUTCSR_1d<lsuTCSR, float>(1);
                    float* b_scalar_ptr = (float*)b_ptr + b_base;
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_scalar_base), (long)b_scalar_ptr);
                    __tx_lsu_sb_bar();
                    __tx_sb_bar();

                    const uint8_t mask = (uint8_t)(broadcast_base | 0b0001);

                    const int chunk0 = (a0 > MTile) ? MTile : a0;
                    config_tcsr_vector<aTCSR>(chunk0);
                    config_tcsr_vector<bTCSR>(1);
                    config_tcsr_vector<cTCSR>(chunk0);
                    configLSUTCSR_1d<lsuTCSR, float>(chunk0);

                    float* a0_ptr = (float*)a_ptr + a_base;
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)a0_ptr);
                    __tx_lsu_sb_bar();
                    __tx_sb_bar();

                    for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                        const int cur_buf = chunk_idx % 2;
                        const int next_buf = (chunk_idx + 1) % 2;
                        const int prev_buf = (chunk_idx - 1 + 2) % 2;

                        // 预取 next A
                        if (chunk_idx < total_chunks - 1) {
                            const int next_size =
                                (a0 - (chunk_idx + 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx + 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(next_size);

                            long next_off = a_base + (long)(chunk_idx + 1) * MTile;
                            float* a_next = (float*)a_ptr + next_off;
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)a_next);
                        }

                        // 回写 prev C
                        if (chunk_idx > 0) {
                            const int prev_size =
                                (a0 - (chunk_idx - 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx - 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(prev_size);

                            long prev_off = c_base + (long)(chunk_idx - 1) * MTile;
                            float* c_prev = (float*)c_ptr + prev_off;
                            tx_lsu_dma_u2d<lsuTCSR>((long)c_prev, (long)uram_ptr(C_base + prev_buf * MTile));
                        }

                        // 计算 cur
                        const int cur_size =
                            (a0 - chunk_idx * MTile > MTile)
                                ? MTile
                                : (a0 - chunk_idx * MTile);

                        config_tcsr_vector<aTCSR>(cur_size);
                        config_tcsr_vector<bTCSR>(1);
                        config_tcsr_vector<cTCSR>(cur_size);

                        DISPATCH_VEC_ADD_FP32(mask,
                            uram_ptr(A_base + cur_buf * MTile),
                            uram_ptr(B_scalar_base),
                            uram_ptr(C_base + cur_buf * MTile));

                        __tx_sb_bar();
                        __tx_lsu_sb_bar();
                    }

                    // 回写最后一个
                    const int last_idx = total_chunks - 1;
                    const int last_buf = last_idx % 2;
                    const int last_size =
                        (a0 - last_idx * MTile > MTile) ? MTile : (a0 - last_idx * MTile);

                    configLSUTCSR_1d<lsuTCSR, float>(last_size);

                    long last_off = c_base + (long)last_idx * MTile;
                    float* c_last = (float*)c_ptr + last_off;
                    tx_lsu_dma_u2d<lsuTCSR>((long)c_last, (long)uram_ptr(C_base + last_buf * MTile));
                    __tx_lsu_sb_bar();

                    continue;
                }

                {
                    const uint8_t mask = broadcast_base;

                    // 预取 chunk0 的 A/B
                    const int chunk0 = (a0 > MTile) ? MTile : a0;
                    config_tcsr_vector<aTCSR>(chunk0);
                    config_tcsr_vector<bTCSR>(chunk0);
                    config_tcsr_vector<cTCSR>(chunk0);
                    configLSUTCSR_1d<lsuTCSR, float>(chunk0);

                    float* a0_ptr = (float*)a_ptr + a_base;
                    float* b0_ptr = (float*)b_ptr + b_base;
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)a0_ptr);
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + 0 * MTile), (long)b0_ptr);
                    __tx_lsu_sb_bar();
                    __tx_sb_bar();

                    for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                        const int cur_buf = chunk_idx % 2;
                        const int next_buf = (chunk_idx + 1) % 2;
                        const int prev_buf = (chunk_idx - 1 + 2) % 2;

                        if (chunk_idx < total_chunks - 1) {
                            const int next_size =
                                (a0 - (chunk_idx + 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx + 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(next_size);

                            long next_a_off = a_base + (long)(chunk_idx + 1) * MTile;
                            long next_b_off = b_base + (long)(chunk_idx + 1) * MTile;
                            float* a_next = (float*)a_ptr + next_a_off;
                            float* b_next = (float*)b_ptr + next_b_off;

                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)a_next);
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)b_next);
                        }

                        if (chunk_idx > 0) {
                            const int prev_size =
                                (a0 - (chunk_idx - 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx - 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(prev_size);

                            long prev_c_off = c_base + (long)(chunk_idx - 1) * MTile;
                            float* c_prev = (float*)c_ptr + prev_c_off;
                            tx_lsu_dma_u2d<lsuTCSR>((long)c_prev, (long)uram_ptr(C_base + prev_buf * MTile));
                        }

                        const int cur_size =
                            (a0 - chunk_idx * MTile > MTile)
                                ? MTile
                                : (a0 - chunk_idx * MTile);

                        config_tcsr_vector<aTCSR>(cur_size);
                        config_tcsr_vector<bTCSR>(cur_size);
                        config_tcsr_vector<cTCSR>(cur_size);

                        DISPATCH_VEC_ADD_FP32(mask,
                            uram_ptr(A_base + cur_buf * MTile),
                            uram_ptr(B_base + cur_buf * MTile),
                            uram_ptr(C_base + cur_buf * MTile));

                        __tx_sb_bar();
                        __tx_lsu_sb_bar();
                    }

                    const int last_idx = total_chunks - 1;
                    const int last_buf = last_idx % 2;
                    const int last_size =
                        (a0 - last_idx * MTile > MTile) ? MTile : (a0 - last_idx * MTile);

                    configLSUTCSR_1d<lsuTCSR, float>(last_size);

                    long last_off = c_base + (long)last_idx * MTile;
                    float* c_last = (float*)c_ptr + last_off;
                    tx_lsu_dma_u2d<lsuTCSR>((long)c_last, (long)uram_ptr(C_base + last_buf * MTile));
                    __tx_lsu_sb_bar();
                }
            }
        }
    }
    return 0;
}

__device__ long kernel_mul_fp32_with_broadcast(void* a_ptr, void* b_ptr, void* c_ptr,
                                                 int a0,int a1,int a2,int a3,
                                                 int b0,int b1,int b2,int b3) {
    const uint8_t broadcast_base = get_broadcast(a0,a1,a2,a3, b0,b1,b2,b3);

    const int a_s1 = a0;
    const int a_s2 = a0 * a1;
    const int a_s3 = a0 * a1 * a2;

    const int b_s1 = b0;
    const int b_s2 = b0 * b1;
    const int b_s3 = b0 * b1 * b2;

    constexpr size_t A_base = 0;
    constexpr size_t B_base = A_base + 2 * MTile;
    constexpr size_t C_base = B_base + 2 * MTile;
    constexpr size_t B_scalar_base = C_base + 2 * MTile;

    for (int i = 0; i < a3; ++i) {
        const int i1 = (b3 == 1) ? 0 : i;
        for (int j = 0; j < a2; ++j) {
            const int j1 = (b2 == 1) ? 0 : j;
            for (int k = 0; k < a1; ++k) {
                const int k1 = (b1 == 1) ? 0 : k;

                const int total_chunks = (a0 + MTile - 1) / MTile;
                if (total_chunks <= 0) continue;

                const long a_base = (long)i * a_s3 + (long)j * a_s2 + (long)k * a_s1;
                const long c_base = a_base;
                const long b_base = (long)i1 * b_s3 + (long)j1 * b_s2 + (long)k1 * b_s1;

                if (b0 == 1) {
                    configLSUTCSR_1d<lsuTCSR, float>(1);
                    float* b_scalar_ptr = (float*)b_ptr + b_base;
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_scalar_base), (long)b_scalar_ptr);
                    __tx_lsu_sb_bar();
                    __tx_sb_bar();

                    const uint8_t mask = (uint8_t)(broadcast_base | 0b0001);

                    const int chunk0 = (a0 > MTile) ? MTile : a0;
                    config_tcsr_vector<aTCSR>(chunk0);
                    config_tcsr_vector<bTCSR>(1);
                    config_tcsr_vector<cTCSR>(chunk0);
                    configLSUTCSR_1d<lsuTCSR, float>(chunk0);

                    float* a0_ptr = (float*)a_ptr + a_base;
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)a0_ptr);
                    __tx_lsu_sb_bar();
                    __tx_sb_bar();

                    for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                        const int cur_buf = chunk_idx % 2;
                        const int next_buf = (chunk_idx + 1) % 2;
                        const int prev_buf = (chunk_idx - 1 + 2) % 2;

                        if (chunk_idx < total_chunks - 1) {
                            const int next_size =
                                (a0 - (chunk_idx + 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx + 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(next_size);

                            long next_off = a_base + (long)(chunk_idx + 1) * MTile;
                            float* a_next = (float*)a_ptr + next_off;
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)a_next);
                        }

                        if (chunk_idx > 0) {
                            const int prev_size =
                                (a0 - (chunk_idx - 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx - 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(prev_size);

                            long prev_off = c_base + (long)(chunk_idx - 1) * MTile;
                            float* c_prev = (float*)c_ptr + prev_off;
                            tx_lsu_dma_u2d<lsuTCSR>((long)c_prev, (long)uram_ptr(C_base + prev_buf * MTile));
                        }

                        const int cur_size =
                            (a0 - chunk_idx * MTile > MTile)
                                ? MTile
                                : (a0 - chunk_idx * MTile);

                        config_tcsr_vector<aTCSR>(cur_size);
                        config_tcsr_vector<bTCSR>(1);
                        config_tcsr_vector<cTCSR>(cur_size);

                        DISPATCH_VEC_MUL_FP32(mask,
                            uram_ptr(A_base + cur_buf * MTile),
                            uram_ptr(B_scalar_base),
                            uram_ptr(C_base + cur_buf * MTile));

                        __tx_sb_bar();
                        __tx_lsu_sb_bar();
                    }

                    const int last_idx = total_chunks - 1;
                    const int last_buf = last_idx % 2;
                    const int last_size =
                        (a0 - last_idx * MTile > MTile) ? MTile : (a0 - last_idx * MTile);

                    configLSUTCSR_1d<lsuTCSR, float>(last_size);

                    long last_off = c_base + (long)last_idx * MTile;
                    float* c_last = (float*)c_ptr + last_off;
                    tx_lsu_dma_u2d<lsuTCSR>((long)c_last, (long)uram_ptr(C_base + last_buf * MTile));
                    __tx_lsu_sb_bar();

                    continue;
                }

                {
                    const uint8_t mask = broadcast_base;

                    const int chunk0 = (a0 > MTile) ? MTile : a0;
                    config_tcsr_vector<aTCSR>(chunk0);
                    config_tcsr_vector<bTCSR>(chunk0);
                    config_tcsr_vector<cTCSR>(chunk0);
                    configLSUTCSR_1d<lsuTCSR, float>(chunk0);

                    float* a0_ptr = (float*)a_ptr + a_base;
                    float* b0_ptr = (float*)b_ptr + b_base;
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + 0 * MTile), (long)a0_ptr);
                    tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + 0 * MTile), (long)b0_ptr);
                    __tx_lsu_sb_bar();
                    __tx_sb_bar();

                    for (int chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
                        const int cur_buf = chunk_idx % 2;
                        const int next_buf = (chunk_idx + 1) % 2;
                        const int prev_buf = (chunk_idx - 1 + 2) % 2;

                        if (chunk_idx < total_chunks - 1) {
                            const int next_size =
                                (a0 - (chunk_idx + 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx + 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(next_size);

                            long next_a_off = a_base + (long)(chunk_idx + 1) * MTile;
                            long next_b_off = b_base + (long)(chunk_idx + 1) * MTile;
                            float* a_next = (float*)a_ptr + next_a_off;
                            float* b_next = (float*)b_ptr + next_b_off;

                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(A_base + next_buf * MTile), (long)a_next);
                            tx_lsu_dma_d2u<lsuTCSR>((long)uram_ptr(B_base + next_buf * MTile), (long)b_next);
                        }

                        if (chunk_idx > 0) {
                            const int prev_size =
                                (a0 - (chunk_idx - 1) * MTile > MTile)
                                    ? MTile
                                    : (a0 - (chunk_idx - 1) * MTile);
                            configLSUTCSR_1d<lsuTCSR, float>(prev_size);

                            long prev_c_off = c_base + (long)(chunk_idx - 1) * MTile;
                            float* c_prev = (float*)c_ptr + prev_c_off;
                            tx_lsu_dma_u2d<lsuTCSR>((long)c_prev, (long)uram_ptr(C_base + prev_buf * MTile));
                        }

                        const int cur_size =
                            (a0 - chunk_idx * MTile > MTile)
                                ? MTile
                                : (a0 - chunk_idx * MTile);

                        config_tcsr_vector<aTCSR>(cur_size);
                        config_tcsr_vector<bTCSR>(cur_size);
                        config_tcsr_vector<cTCSR>(cur_size);

                        DISPATCH_VEC_MUL_FP32(mask,
                            uram_ptr(A_base + cur_buf * MTile),
                            uram_ptr(B_base + cur_buf * MTile),
                            uram_ptr(C_base + cur_buf * MTile));

                        __tx_sb_bar();
                        __tx_lsu_sb_bar();
                    }

                    const int last_idx = total_chunks - 1;
                    const int last_buf = last_idx % 2;
                    const int last_size =
                        (a0 - last_idx * MTile > MTile) ? MTile : (a0 - last_idx * MTile);

                    configLSUTCSR_1d<lsuTCSR, float>(last_size);

                    long last_off = c_base + (long)last_idx * MTile;
                    float* c_last = (float*)c_ptr + last_off;
                    tx_lsu_dma_u2d<lsuTCSR>((long)c_last, (long)uram_ptr(C_base + last_buf * MTile));
                    __tx_lsu_sb_bar();
                }
            }
        }
    }
    return 0;
}

__device__ long kernel_add_fp32(void* a_ptr, void* b_ptr, void* c_ptr,
                                  int a0,int a1,int a2,int a3,
                                  int b0,int b1,int b2,int b3) {
    if (!needs_broadcast(a0,a1,a2,a3, b0,b1,b2,b3)) {
        return kernel_add_fp32_without_broadcast(a_ptr,b_ptr,c_ptr, a0,a1,a2,a3);
    }
    return kernel_add_fp32_with_broadcast(a_ptr,b_ptr,c_ptr, a0,a1,a2,a3, b0,b1,b2,b3);
}

__device__ long kernel_mul_fp32(void* a_ptr, void* b_ptr, void* c_ptr,
                                  int a0,int a1,int a2,int a3,
                                  int b0,int b1,int b2,int b3) {
    if (!needs_broadcast(a0,a1,a2,a3, b0,b1,b2,b3)) {
        return kernel_mul_fp32_without_broadcast(a_ptr,b_ptr,c_ptr, a0,a1,a2,a3);
    }
    return kernel_mul_fp32_with_broadcast(a_ptr,b_ptr,c_ptr, a0,a1,a2,a3, b0,b1,b2,b3);
}

#undef DISPATCH_VEC_ADD_FP32
#undef DISPATCH_VEC_MUL_FP32
