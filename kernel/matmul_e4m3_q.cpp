#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

template <long aTCFG, long bTCFG, long cTCFG, long aTileTCFG, long bTileTCFG, long cTileTCFG, bool A_T>
static inline void updateMTile(long mTile)
{
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(mTile);
    if (!A_T)
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(mTile);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(mTile);
}

template <long aTCFG, long bTCFG, long cTCFG, long aTileTCFG, long bTileTCFG, long cTileTCFG, bool A_T>
static inline void updateNTile(long nTile)
{
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM1>(nTile);
}

template <long aTCFG, long bTCFG, long cTCFG, long aTileTCFG, long bTileTCFG, long cTileTCFG, bool A_T>
static inline void updateKTile(long kTile)
{
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(kTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM0>(kTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE1>(kTile);
}


#define log2ElementSizeA (0)
#define log2ElementSizeB (0)
#define log2ElementSizeC (2)
#define MaxMTile (32)
#define MaxNTile (32)

float kernel_buffer[MaxMTile * MaxNTile];
template <long mTile, long nTile, long kTile, long Q_K>
__attribute__((always_inline)) static inline long gmma_e4m3q_fp32_AT(void *a, void *b, void *c,
                                                                     int m, int n, int k,
                                                                     int lda, int ldb, int ldc)
{ /* Setup TCFGs for compute */
    if ((Q_K % kTile) != 0)
    {
        printf("Not supported: Q_K % kTile != 0\n");
        return -1;
    }
    if ((k % Q_K) != 0)
    {
        printf("Not supported: k % Q_K != 0\n");
        return -1;
    }
    if(Q_K != kTile) {
        printf("Not supported: Q_K != kTile\n");
        return -1;
    }

    const long aTCFG = 0;
    const long bTCFG = 1;
    const long cTCFG = 2;
    const long aTileTCFG = 3;
    const long bTileTCFG = 4;
    const long cTileTCFG = 5;
    /* __tx_tcsr_write is alias for __tx_tcsr_xchg_nmask */
    /* Mem TCFGs */
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(lda);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(kTile);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(ldb);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(ldc);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);

    /* NRAM TCFGs */
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(kTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM0>(kTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE1>(kTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM1>(nTile);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(mTile);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM3>(1);

    /* Scratchpad for the tile */
    void *aTile = 0;
    void *bTile = (void *)((long)aTile + mTile * kTile);
    void *cPartitialTile = (void *)((long)bTile + kTile * nTile);
    // void *cTile = (void *)((long)cPartitialTile + mTile * nTile * sizeof(float));
    float* cF = (float*)c;

    for (long m_i = 0; m_i < m; m_i += mTile)
    {
        long m_left = m - m_i;
        int realMTile = mTile;
        if (m_left < mTile)
        {
            realMTile = m_left;
        }
        updateMTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, true>(realMTile);
        for (long n_i = 0; n_i < n; n_i += nTile)
        {
            long n_left = n - n_i;
            int realNTile = nTile;
            if (n_left < nTile)
            {
                realNTile = n_left; 
            }
            updateNTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, true>(realNTile);
            __tx_fence(); /* Load C */
            // __tx_async_nram_load<cTCFG, cTileTCFG, log2ElementSizeC>((void *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)), cTile);
            for (long k_ii = 0; k_ii < k; k_ii += Q_K)
            {
                long k_ii_offset = (k_ii / Q_K);
                memset(kernel_buffer, 0, sizeof(float) * realMTile * realNTile);
                __tx_fence();
                __tx_async_nram_load<cTileTCFG, cTileTCFG, log2ElementSizeC>(kernel_buffer, cPartitialTile);
                // Partital C is computed in each Q_K x Q_K block
                for (long k_i = k_ii; k_i < k_ii + Q_K; k_i += kTile)
                {
                    long real_k_i = k_i + (1 + k_ii_offset) * sizeof(float);
                    __tx_async_nram_load<aTCFG, aTileTCFG, log2ElementSizeA>((void *)((long)a + (lda * m_i + real_k_i) * (1 << log2ElementSizeA)), aTile);
                    __tx_async_nram_load<bTCFG, bTileTCFG, log2ElementSizeB>((void *)((long)b + (ldb * n_i + real_k_i) * (1 << log2ElementSizeB)), bTile);
                    __tx_fence(); /* Compute C = C + A x B */
                    __tx_matmul_subfloat_e4m3_e4m3_f32<aTileTCFG, bTileTCFG, cTileTCFG>(aTile, bTile, cPartitialTile);
                    __tx_fence();
                }
                __tx_async_nram_store<cTileTCFG, cTileTCFG, log2ElementSizeC>(cPartitialTile, kernel_buffer);
                __tx_fence();
                // Scale for the C tile
                for(int m_ii = 0 ; m_ii < realMTile; m_ii += 1) {
                    for(int n_ii = 0 ; n_ii < realNTile; n_ii += 1) {
                        long real_m = m_i + m_ii;
                        long real_n = n_i + n_ii;
                        float scale_a = *(float *)((long)a + (lda * real_m + k_ii * 1 + k_ii_offset * sizeof(float)));
                        float scale_b = *(float *)((long)b + (ldb * real_n + k_ii * 1 + k_ii_offset * sizeof(float)));
                        float scale = scale_a * scale_b;
                        cF[ldc * real_n + real_m] += 1.0f + kernel_buffer[m_ii * realNTile + n_ii] * scale;
                        if(real_m == 0 && real_n == 0) printf("cF[m:%d,n:%d] = %x\n", real_m, real_n, *(int*)&cF[ldc * real_n + real_m]);
                    }
                }
            }
            __tx_fence();
            // __tx_async_nram_store<cTileTCFG, cTCFG, log2ElementSizeC>(cTile, (void *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)));
        }
    }
    __tx_fence();
    return 0;
}

__device__ long gmma_e4m3q_fp32_AT_QK32(void *a, void *b, void *c,
                                              int m, int n, int k,
                                              int lda, int ldb, int ldc)
{
    return gmma_e4m3q_fp32_AT<32, 32, 32, 32>(a, b, c, m, n, k, lda, ldb, ldc);
}

__device__ long gmma_e4m3q_fp32_AT_QK256(void *a, void *b, void *c,
                                              int m, int n, int k,
                                              int lda, int ldb, int ldc)
{
    return gmma_e4m3q_fp32_AT<32, 32, 256, 256>(a, b, c, m, n, k, lda, ldb, ldc);
}
