#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_cache.h"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <vector>
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long dTCSR = 3;
const long oneTCSR = 4;
const long lsuTCSR = 0;

extern int _uram[256 * 1024];

template <int TCSR_ID> static inline void config_tcsr_vector(int m) {
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
static inline void configLSUTCSR(int m, int n) {

    // NOTE: 新硬件配置寄存器 Stride 数值均以 Bytes 为单位
    long m_bytes = m * sizeof(elemty);
    if(n && ((m_bytes % 64) != 0)) {
        printf("LSU transfer size must be aligned to 64B, but got m_bytes=%ld\n", m_bytes);
    }

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(((m_bytes + 63) / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(n - 1);
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

#define MAX_VEC_LEN 0x1000
static float theta_interp_cache[0x1000] __attribute__((aligned(64)));;

static inline void rope_yarn(uint64_t cos_cache_uram, uint64_t sin_cache_uram, float theta_base,
                             float theta_scale, float freq_scale, int m,
                             float *corr_dims = NULL,   // useless in qwen2
                             float *freq_factor = NULL, // freq_factor is always NULL in qwen2
                             float ext_factor = 0.0f,   // // ext_factor is always 0.0f in qwen2
                             float mscale = 1.0f,  // mscale == attn_factor is always 1.0f in qwen2
                             float sin_sign = 1.0f // sin_sign is always 1.0f in forward computing
) {
    float *sp = (float *)_uram;
    float *rama = (float *)_uram + 0x1000;
    float *ramb = (float *)_uram + 0x2000;
    float *ramc = (float *)_uram + 0x3000;

    float *ram_mscale = sp;
    float *ram_sin_sign = sp + 0x800;
    *ram_mscale = mscale;
    *ram_sin_sign = sin_sign;

    float theta = theta_base;
    for (int i = 0; i < m / 2; i++) {
        const float ff = 1.0f;
        float theta_extrap = theta / ff;
        theta_interp_cache[i] = freq_scale * theta_extrap;
        theta *= theta_scale;
    }
    cache_flush_range(theta_interp_cache, (m / 2) * sizeof(theta_interp_cache[0]));
    tx_lsu_dma_d2u<lsuTCSR>((long)rama, (long)theta_interp_cache);
    __tx_lsu_sb_bar();

    __tx_sfu_cos_fp32<aTCSR, aTCSR>(rama, ramb);
    __tx_sfu_sin_fp32<aTCSR, aTCSR>(rama, ramc);
    __tx_fp_mul_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>((void *)ramb, (void *)ram_mscale, (void *)cos_cache_uram); // cos * mscale
    __tx_fp_mul_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>((void *)ramc, (void *)ram_mscale, (void *)ramc); // sin * mscale
    __tx_fp_mul_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>((void *)ramc, (void *)ram_mscale, (void *)sin_cache_uram); // sin * mscale * sin_sign
    __tx_sb_bar();
}

__device__ long rope_f32_i32_f32(float *a, int *b, float *res, int bs, int h, int n, int m,
                                 int n_dims, uint32_t theta_scale_, uint32_t freq_scale_,
                                 int mdst) {
    uint64_t uram0 = (uint64_t)((char *)_uram + 0x01000);
    uint64_t uram1 = (uint64_t)((char *)_uram + 0x02000);
    uint64_t uram2 = (uint64_t)((char *)_uram + 0x03000);
    uint64_t uram3 = (uint64_t)((char *)_uram + 0x04000);
    uint64_t uram4 = (uint64_t)((char *)_uram + 0x05000);
    uint64_t uram5 = (uint64_t)((char *)_uram + 0x06000);
    uint64_t uram6 = (uint64_t)((char *)_uram + 0x07000);
    uint64_t uram7 = (uint64_t)((char *)_uram + 0x08000);
    float theta_scale = u32_as_float(theta_scale_);
    float freq_scale = u32_as_float(freq_scale_);
    const int vec_len = n_dims / 2;

    if (vec_len > MAX_VEC_LEN) {
        printf("rope: n_dims is too large: %x(> MAX_VEC_LEN=%x)\n", vec_len, MAX_VEC_LEN);
        return -1;
    }

    config_tcsr_vector<oneTCSR>(1);
    config_tcsr_vector<aTCSR>(vec_len);
    config_tcsr_vector<bTCSR>(vec_len);
    config_tcsr_vector<cTCSR>(vec_len);
    config_tcsr_vector<dTCSR>(vec_len);

    configLSUTCSR<lsuTCSR, float>(vec_len, 1);

    // The tail of a (beyond n_dims) and the position index b are both read directly by the CPU;
    // these buffers are written by DMA/host, so invalidate before reading to avoid stale L1 cache
    // lines.
    cache_inval_range(a, (size_t)bs * h * n * m * sizeof(float));
    cache_inval_range(b, (size_t)h * sizeof(*b));

    for (int i = 0; i < bs; ++i) {    // batch
        for (int j = 0; j < h; ++j) { // seq-len
            int64_t p = b[j];
            rope_yarn(uram2, uram3, p, theta_scale, freq_scale, mdst);

            for (int k = 0; k < n; ++k) { // attn-heads
                const int64_t head_offset = (i * h * n + j * n + k) * m;
                float *x0_ptr = &a[head_offset];
                float *x1_ptr = &a[head_offset + vec_len];
                float *res0_ptr = &res[head_offset];
                float *res1_ptr = &res[head_offset + vec_len];

                // load stage
                tx_lsu_dma_d2u<lsuTCSR>((long)uram0, (long)x0_ptr);
                tx_lsu_dma_d2u<lsuTCSR>((long)uram1, (long)x1_ptr);
                __tx_lsu_sb_bar();

                // compute stage
                __tx_fp_mul_fp32<aTCSR, cTCSR, aTCSR, 0b0000>((void *)uram0, (void *)uram2,
                                                              (void *)uram4); // buf1 = x0 * cos
                __tx_fp_mul_fp32<bTCSR, dTCSR, bTCSR, 0b0000>((void *)uram1, (void *)uram3,
                                                              (void *)uram5); // buf2 = x1 * sin
                __tx_fp_mul_fp32<aTCSR, dTCSR, aTCSR, 0b0000>((void *)uram0, (void *)uram3,
                                                              (void *)uram6); // buf3 = x0 * sin
                __tx_fp_mul_fp32<bTCSR, cTCSR, bTCSR, 0b0000>((void *)uram1, (void *)uram2,
                                                              (void *)uram7); // buf4 = x1 * cos
                __tx_sb_bar();

                __tx_fp_sub_fp32<aTCSR, aTCSR, aTCSR, 0b0000>((void *)uram4, (void *)uram5,
                                                              (void *)uram0); // res0 = buf1 - buf2
                __tx_fp_add_fp32<aTCSR, aTCSR, aTCSR, 0b0000>((void *)uram6, (void *)uram7,
                                                              (void *)uram1); // res1 = buf3 + buf4
                __tx_sb_bar();

                // store stage
                // fill res[i][j][k][:prope.n_dims]
                tx_lsu_dma_u2d<lsuTCSR>((long)res0_ptr, (long)uram0);
                tx_lsu_dma_u2d<lsuTCSR>((long)res1_ptr, (long)uram1);
                __tx_lsu_sb_bar();

                // fill remain res[i][j][k][prope.n_dims:]
                const int64_t fill_res_idx = i * h * n * m + j * n * m + k * m;
                for (int p = n_dims; p < m; p += 2) {
                    res[fill_res_idx + p] = a[fill_res_idx + p];
                    res[fill_res_idx + p + 1] = a[fill_res_idx + p + 1];
                }
                cache_flush_range(&res[fill_res_idx], sizeof(res[fill_res_idx]) * m);
            }
        }
    }

    return 0;
}
