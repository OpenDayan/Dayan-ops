#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <math.h>
#include <stdio.h>
#include <string.h>
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long oneTCSR = 3;
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
    long m_bytes = m * sizeof(elemty);
    m_bytes = m_bytes < 64 ? 64 : m_bytes;
    if(n && ((m_bytes % 64) != 0)) {
        // TODO: 不支持非对齐传输
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

__device__ long softmax_f32_f32_f32(float *a, float *b, float *res, uint32_t scale_, int bs, int h,
                                    int n, int m) {
    float scale = u32_as_float(scale_);
    float *sp = (float *)_uram;
    float *scale_ram = sp + 0x100;
    float *slope_ram = sp + 0x200;
    float *max_ram = sp + 0x300;
    float *sum_ram = sp + 0x400;

    void * uram0 = (void *)((char *)_uram + 0x10000);
    void * uram1 = (void *)((char *)_uram + 0x20000);

    config_tcsr_vector<aTCSR>(m);
    config_tcsr_vector<bTCSR>(m);
    config_tcsr_vector<cTCSR>(m);
    config_tcsr_vector<oneTCSR>(1);
    configLSUTCSR<lsuTCSR, float>(m, 1);

    float slope = 1.0f; // only support max_bias == 0.0f in qwen2
    *scale_ram = scale;
    *slope_ram = slope;

    for (int i = 0; i < bs; ++i) {          // batch
        for (int j = 0; j < (h * n); ++j) { // attn-heads * seq_len
            const uint32_t hidx = (j / n);  // current head idx
            float *a_ptr = &a[i * h * n * m + j * m];
            float *b_ptr = &b[i * h * n * m + (j % n) * m];
            float *res_ptr = &res[i * h * n * m + j * m];

            // load stage
            tx_lsu_dma_d2u<lsuTCSR>((long)uram0, (long)a_ptr);
            tx_lsu_dma_d2u<lsuTCSR>((long)uram1, (long)b_ptr);
            __tx_lsu_sb_bar();

            // compute stage
            __tx_fp_mul_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>(
                (void *)uram0, scale_ram, (void *)uram0); // scale: a[i][j][k][:] *= scale
            __tx_fp_mul_fp32<bTCSR, oneTCSR, cTCSR, 0b0001>(
                (void *)uram1, slope_ram, (void *)uram1); // mask:  temp = slope * b[i][0][k][:]
            __tx_fp_add_fp32<aTCSR, cTCSR, aTCSR, 0b0000>(
                (void *)uram0, (void *)uram1, (void *)uram0); // mask: a[i][j][k][:] += temp
            __tx_fp_reduce_max_fp32<aTCSR, oneTCSR, 0>((void *)uram0,
                                                       max_ram); // maxval = max(a[i][j][k][:])
            __tx_fp_sub_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>(
                (void *)uram0, (void *)max_ram,
                (void *)uram0); // a[i][j][k][:] = a[i][j][k][:] - max
            __tx_sfu_exp_fp32<aTCSR, bTCSR>((void *)uram0,
                                            (void *)uram1); // exp_temp = exp(a[i][j][k][:])
            __tx_fp_reduce_add_fp32<aTCSR, oneTCSR, 0>((void *)uram1,
                                                       (void *)sum_ram); // sum = sum of exp_temp
            __tx_sfu_recip_fp32<oneTCSR, oneTCSR>((void *)sum_ram, (void *)sum_ram); // sum_ram = 1/sum

            __tx_fp_mul_fp32<bTCSR, oneTCSR, cTCSR, 0b0001>((void *)uram1, sum_ram,
                                                            (void *)uram1); // exp_temp *= 1/sum
            __tx_sb_bar();

            // store stage
            tx_lsu_dma_u2d<lsuTCSR>((long)res_ptr, (long)uram1);
            __tx_lsu_sb_bar();
        }
    }

    return 0;
}