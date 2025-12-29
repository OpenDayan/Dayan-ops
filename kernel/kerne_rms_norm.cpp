#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <cmath>
#include <stdio.h>
#include <string.h>
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long oneTCFG = 3;
const long lsuTCSR = 0;

static inline void configTCSRs(int m, int n) {
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(m * n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(m * n);

    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(m * n);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(m * n);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(m * n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(m * n);
}

static inline void configoneTCSR(void) {
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE2>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE3>(1);
}

template <long GROUPBASE, typename elemty>
static inline void configLSUTCSR(int m, int n) {
    long m_bytes = m * sizeof(elemty);
    m_bytes = m_bytes < 64 ? 64 : m_bytes;
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

// NOTE: 新版硬件 URAM 全局符号
extern int _uram[256 * 1024];

__device__ long rms_norm_f32(float *x, float *y, int b, int h, int n, int m, int epsilon) {
    const int64_t total_rows = b * h * n;
    float m0_val = 1.0f / (float)m;

    int op_count = total_rows;

    float *XTile[2] = {(float *)_uram, (float *)_uram + m};
    float *YTile[2] = {XTile[1] + m, XTile[1] + 2 * m};
    float *SquareTile = YTile[1] + m;
    float *RowMean = SquareTile + m;
    float *InvStd = RowMean + 1;
    float *M0Ram = InvStd + 1;
    float *EpsilonRam = M0Ram + 1;

    configoneTCSR();
    configTCSRs(m, 1);

    *M0Ram = m0_val;
    *(int*)EpsilonRam = epsilon; 

    if (op_count == 1) { // 数据足以单次处理
        configTCSRs(m, 1);
        configLSUTCSR<lsuTCSR, float>(m, 1);

        // D2U: Load x
        tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[0]), (long)x);
        __tx_lsu_sb_bar(); // 等待数据加载完成

        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0>(XTile[0], XTile[0], SquareTile);
        __tx_fp_reduce_add_fp32<aTCSR, oneTCFG, 0>(SquareTile, RowMean);
        __tx_fp_mul_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, M0Ram, RowMean);
        __tx_fp_add_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, EpsilonRam, RowMean);
        __tx_sfu_rsqrt_fp32<oneTCFG, oneTCFG>(RowMean, InvStd);
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[0], InvStd, YTile[0]);
        __tx_sb_bar(); // 等待计算完成

        tx_lsu_dma_u2d<lsuTCSR>((long)y, (long)(YTile[0]));
        __tx_lsu_sb_bar();

    } else {
        // load_0
        configTCSRs(m, 1);
        configLSUTCSR<lsuTCSR, float>(m, 1); 

        tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[0]), (long)x);
        __tx_lsu_sb_bar(); 
        __tx_sb_bar();

        // compute_0
        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0>(XTile[0], XTile[0], SquareTile);
        __tx_fp_reduce_add_fp32<aTCSR, oneTCFG, 0>(SquareTile, RowMean);
        __tx_fp_mul_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, M0Ram, RowMean);
        __tx_fp_add_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, EpsilonRam, RowMean);
        __tx_sfu_rsqrt_fp32<oneTCFG, oneTCFG>(RowMean, InvStd);
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[0], InvStd, YTile[0]);

        // load_1
        tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[1]), (long)(&x[m]));
        __tx_lsu_sb_bar();
        __tx_sb_bar();

        for (int i = 2; i < op_count; i++) {
            int store_num = i - 2;
            int compute_num = i - 1;
            int load_num = i;

            int store_buffer_num = store_num % 2;
            int compute_buffer_num = compute_num % 2;
            int load_buffer_num = load_num % 2;

            configLSUTCSR<lsuTCSR, float>(m, 1);

            // store_i-2 (U2D)
            float *y_store = y + store_num * m;
            tx_lsu_dma_u2d<lsuTCSR>((long)y_store, (long)(YTile[store_buffer_num]));

            // compute_i-1
            __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0>(
                XTile[compute_buffer_num], XTile[compute_buffer_num], SquareTile);
            __tx_fp_reduce_add_fp32<aTCSR, oneTCFG, 0>(SquareTile, RowMean);
            __tx_fp_mul_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, M0Ram, RowMean);
            __tx_fp_add_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, EpsilonRam, RowMean);
            __tx_sfu_rsqrt_fp32<oneTCFG, oneTCFG>(RowMean, InvStd);
            __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[compute_buffer_num], InvStd,
                                                            YTile[compute_buffer_num]);

            // load_i (D2U)
            float *x_load = x + load_num * m;
            tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[load_buffer_num]), (long)x_load);
            
            __tx_lsu_sb_bar();
            __tx_sb_bar();
        }

        // store_n-1
        float *y_store_0 = y + (op_count - 2) * m;
        tx_lsu_dma_u2d<lsuTCSR>((long)y_store_0, (long)(YTile[op_count % 2]));

        // compute_n
        int compute_buffer_num = (op_count - 1) % 2;
        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0>(XTile[compute_buffer_num],
                                                 XTile[compute_buffer_num], SquareTile);
        __tx_fp_reduce_add_fp32<aTCSR, oneTCFG, 0>(SquareTile, RowMean);
        __tx_fp_mul_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, M0Ram, RowMean);
        __tx_fp_add_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(RowMean, EpsilonRam, RowMean);
        __tx_sfu_rsqrt_fp32<oneTCFG, oneTCFG>(RowMean, InvStd);
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[compute_buffer_num], InvStd,
                                                        YTile[compute_buffer_num]);
        __tx_sb_bar();

        // store_n
        float *y_store_1 = y + (op_count - 1) * m;
        tx_lsu_dma_u2d<lsuTCSR>((long)y_store_1, (long)(YTile[(op_count - 1) % 2]));
        __tx_lsu_sb_bar();
    }

    return 0;
}