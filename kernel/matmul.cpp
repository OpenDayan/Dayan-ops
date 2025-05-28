#include "npu_instruction.hpp"

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
        __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(A_T ? kTile : 1);                                                                                                           \
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
            long m_left = m - m_i;                                                                                                                                                \
            if (m_left < mTile)                                                                                                                                                   \
            {                                                                                                                                                                     \
                updateMTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, A_T>(m_left);                                                                                   \
            }                                                                                                                                                                     \
            for (long n_i = 0; n_i < n; n_i += nTile)                                                                                                                             \
            {                                                                                                                                                                     \
                long n_left = n - n_i;                                                                                                                                            \
                if (n_left < nTile)                                                                                                                                               \
                {                                                                                                                                                                 \
                    updateNTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, A_T>(n_left);                                                                               \
                }                                                                                                                                                                 \
                __tx_fence();                                                                                                                                                     \
                /* Load C */                                                                                                                                                      \
                __tx_async_nram_load<cTCFG, cTileTCFG, log2ElementSizeC>((void *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)), cTile);                                 \
                for (long k_i = 0; k_i < k; k_i += kTile)                                                                                                                         \
                {                                                                                                                                                                 \
                    long k_left = k - k_i;                                                                                                                                        \
                    if (k_left < kTile)                                                                                                                                           \
                    {                                                                                                                                                             \
                        updateKTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, A_T>(k_left);                                                                           \
                    }                                                                                                                                                             \
                    /* load A */                                                                                                                                                  \
                    __tx_async_nram_load<aTCFG, aTileTCFG, log2ElementSizeA>((void *)((long)a + (A_T ? (lda * m_i + k_i) : (lda * k_i + m_i)) * (1 << log2ElementSizeA)), aTile); \
                    /* load B */                                                                                                                                                  \
                    __tx_async_nram_load<bTCFG, bTileTCFG, log2ElementSizeB>((void *)((long)b + (ldb * n_i + k_i) * (1 << log2ElementSizeB)), bTile);                             \
                    __tx_fence();                                                                                                                                                 \
                    /* Compute C = C + A x B */                                                                                                                                   \
                    matmulFunc<aTileTCFG, bTileTCFG, cTileTCFG>(aTile, bTile, cTile);                                                                                             \
                    __tx_fence();                                                                                                                                                 \
                    if (k_left < kTile)                                                                                                                                           \
                    {                                                                                                                                                             \
                        updateKTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, A_T>(kTile);                                                                           \
                    }                                                                                                                                                             \
                }                                                                                                                                                                 \
                /* Store C */                                                                                                                                                     \
                __tx_async_nram_store<cTileTCFG, cTCFG, log2ElementSizeC>(cTile, (void *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)));                                \
                if (n_left < nTile)                                                                                                                                               \
                {                                                                                                                                                                 \
                    updateNTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, A_T>(nTile);                                                                                \
                }                                                                                                                                                                 \
            }                                                                                                                                                                     \
        }                                                                                                                                                                         \
        __tx_fence();                                                                                                                                                             \
        return 0;                                                                                                                                                                 \
    }

GEN_MATRIX_KERNEL(gmma_e4m3_e4m3_f32, 0, 0, 2, __tx_matmul_subfloat_e4m3_e4m3_f32)
GEN_MATRIX_KERNEL(gmma_e4m3_e4m3_bf16, 0, 0, 1, __tx_matmul_subfloat_e4m3_e4m3_bf16)
GEN_MATRIX_KERNEL(gmma_s8_s8_s32, 0, 0, 2, __tx_matmul_fix_s8_s8_s32)

extern "C" long gmma_e4m3_f32_16_16_16(void *a, void *b, void *c,
    int m, int n, int k,
    int lda, int ldb, int ldc)
{
return gmma_e4m3_e4m3_f32<16, 16, 16, false>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_e4m3_bf16_16_16_16(void *a, void *b, void *c,
                                        int m, int n, int k,
                                        int lda, int ldb, int ldc)
{
    return gmma_e4m3_e4m3_bf16<16, 16, 16, false>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_s8_s32_16_16_16(void *a, void *b, void *c,
                                     int m, int n, int k,
                                     int lda, int ldb, int ldc)
{
    return gmma_s8_s8_s32<16, 16, 16, false>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_e4m3_bf16_16_16_16_AT(void *a, void *b, void *c,
                                           int m, int n, int k,
                                           int lda, int ldb, int ldc)
{
    return gmma_e4m3_e4m3_bf16<16, 16, 16, true>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_e4m3_f32_16_16_16_AT(void *a, void *b, void *c,
                                           int m, int n, int k,
                                           int lda, int ldb, int ldc)
{
    return gmma_e4m3_e4m3_f32<16, 16, 16, true>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_s8_s32_16_16_16_AT(void *a, void *b, void *c,
                                        int m, int n, int k,
                                        int lda, int ldb, int ldc)
{
    return gmma_s8_s8_s32<16, 16, 16, true>(a, b, c, m, n, k, lda, ldb, ldc);
}
