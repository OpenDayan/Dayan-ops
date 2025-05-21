#include "npu_instruction.hpp"

// Note: A, B, C are column-major matrices
// A **CAN BE** Transposed
// A[k][lda]
// B[n][ldb]
// C[n][ldc]
// C = C + A(^T) x B
// Note: C^T = C^T + B^T x A(^T)
#define GEN_MATRIX_KERNEL(name, log2ElementSizeA, log2ElementSizeB, log2ElementSizeC, matmulFunc)                                                                                 \
    template <long mTile, long nTile, long kTile, bool A_T>                                                                                                                       \
    __attribute__((always_inline)) static inline long name(void *a, void *b, void *c,                                                                                             \
                                                           int m, int n, int k,                                                                                                   \
                                                           int lda, int ldb, int ldc)                                                                                             \
    {                                                                                                                                                                             \
        if (m % mTile != 0 || n % nTile != 0 || k % kTile != 0)                                                                                                                   \
        {                                                                                                                                                                         \
            return -1;                                                                                                                                                            \
        }                                                                                                                                                                         \
        /* Setup TCFGs for compute */                                                                                                                                             \
        const long aTCFG = 0;                                                                                                                                                     \
        const long bTCFG = 1;                                                                                                                                                     \
        const long cTCFG = 2;                                                                                                                                                     \
        const long aTileTCFG = 3;                                                                                                                                                 \
        const long bTileTCFG = 4;                                                                                                                                                 \
        const long cTileTCFG = 5;                                                                                                                                                 \
        /* __tx_tcsr_write is alias for __tx_tcsr_xchg_nmask */                                                                                                                   \
        /* Mem TCFGs */                                                                                                                                                           \
        __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(mTile);                                                                                                                            \
        __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(A_T ? lda : 1);                                                                                                                 \
        __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(kTile);                                                                                                                            \
        __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(A_T ? 1 : lda);                                                                                                                 \
        __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);                                                                                                                                \
        __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);                                                                                                                                \
                                                                                                                                                                                  \
        __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(kTile);                                                                                                                            \
        __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);                                                                                                                             \
        __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(nTile);                                                                                                                            \
        __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(ldb);                                                                                                                           \
        __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);                                                                                                                                \
        __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);                                                                                                                                \
                                                                                                                                                                                  \
        __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(mTile);                                                                                                                            \
        __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);                                                                                                                             \
        __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(nTile);                                                                                                                            \
        __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(ldc);                                                                                                                           \
        __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);                                                                                                                                \
        __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);                                                                                                                                \
                                                                                                                                                                                  \
        /* NRAM TCFGs */                                                                                                                                                          \
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(mTile);                                                                                                                        \
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(A_T ? mTile : 1);                                                                                                           \
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(kTile);                                                                                                                        \
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(A_T ? 1 : mTile);                                                                                                           \
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM2>(1);                                                                                                                            \
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM3>(1);                                                                                                                            \
                                                                                                                                                                                  \
        __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM0>(kTile);                                                                                                                        \
        __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE0>(1);                                                                                                                         \
        __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM1>(nTile);                                                                                                                        \
        __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE1>(kTile);                                                                                                                     \
        __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM2>(1);                                                                                                                            \
        __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM3>(1);                                                                                                                            \
                                                                                                                                                                                  \
        __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM0>(mTile);                                                                                                                        \
        __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE0>(1);                                                                                                                         \
        __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM1>(nTile);                                                                                                                        \
        __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(mTile);                                                                                                                     \
        __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM2>(1);                                                                                                                            \
        __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM3>(1);                                                                                                                            \
                                                                                                                                                                                  \
        /* Scratchpad for the tile */                                                                                                                                             \
        void *aTile = 0;                                                                                                                                                          \
        void *bTile = (void *)((long)aTile + mTile * kTile);                                                                                                                      \
        void *cTile = (void *)((long)bTile + kTile * nTile);                                                                                                                      \
                                                                                                                                                                                  \
        for (long m_i = 0; m_i < m; m_i += mTile)                                                                                                                                 \
        {                                                                                                                                                                         \
            for (long n_i = 0; n_i < n; n_i += nTile)                                                                                                                             \
            {                                                                                                                                                                     \
                __tx_fence();                                                                                                                                                     \
                /* Load C */                                                                                                                                                      \
                __tx_async_nram_load<cTCFG, cTileTCFG, log2ElementSizeC>((void *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)), cTile);                                 \
                for (long k_i = 0; k_i < k; k_i += kTile)                                                                                                                         \
                {                                                                                                                                                                 \
                    /* load A */                                                                                                                                                  \
                    __tx_async_nram_load<aTCFG, aTileTCFG, log2ElementSizeA>((void *)((long)a + (A_T ? (lda * m_i + k_i) : (lda * k_i + m_i)) * (1 << log2ElementSizeA)), aTile); \
                    /* load B */                                                                                                                                                  \
                    __tx_async_nram_load<bTCFG, bTileTCFG, log2ElementSizeB>((void *)((long)b + (ldb * n_i + k_i) * (1 << log2ElementSizeB)), bTile);                             \
                    __tx_fence();                                                                                                                                                 \
                    /* Compute C = C + A x B */                                                                                                                                   \
                    matmulFunc<aTileTCFG, bTileTCFG, cTileTCFG>(aTile, bTile, cTile);                                                                                             \
                    __tx_fence();                                                                                                                                                 \
                }                                                                                                                                                                 \
                /* Store C */                                                                                                                                                     \
                __tx_async_nram_store<cTileTCFG, cTCFG, log2ElementSizeC>(cTile, (void *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)));                                \
            }                                                                                                                                                                     \
        }                                                                                                                                                                         \
        __tx_fence();                                                                                                                                                             \
        return 0;                                                                                                                                                                 \
    }

GEN_MATRIX_KERNEL(gmma_e4m3_e4m3_bf16, 0, 0, 1, __tx_matmul_subfloat_e4m3_e4m3_bf16)
GEN_MATRIX_KERNEL(gmma_s8_s8_s32, 0, 0, 2, __tx_matmul_fix_s8_s8_s32)

extern "C" long gmma_e4m3_bf16_32_32_32(void *a, void *b, void *c,
                                        int m, int n, int k,
                                        int lda, int ldb, int ldc)
{
    return gmma_e4m3_e4m3_bf16<32, 32, 32, false>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_s8_s32_32_32_32(void *a, void *b, void *c,
                                     int m, int n, int k,
                                     int lda, int ldb, int ldc)
{
    return gmma_s8_s8_s32<32, 32, 32, false>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_e4m3_bf16_32_32_32_AT(void *a, void *b, void *c,
    int m, int n, int k,
    int lda, int ldb, int ldc)
{
return gmma_e4m3_e4m3_bf16<32, 32, 32, true>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_s8_s32_32_32_32_AT(void *a, void *b, void *c,
 int m, int n, int k,
 int lda, int ldb, int ldc)
{
return gmma_s8_s8_s32<32, 32, 32, true>(a, b, c, m, n, k, lda, ldb, ldc);
}

