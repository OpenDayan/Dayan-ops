#include "npu_instruction.hpp"

// Note: A, B, C are column-major matrices
// A[k][lda]
// B[n][ldb]
// C[n][ldc]
// C = C + A x B
// Note: C^T = C^T + B^T x A^T
template<long mTile, long nTile, long kTile>
__attribute__((always_inline))
static inline long gmma_e4m3_bf16(void* a, void* b, void* c,
                   int m,   int n,   int k,
                   int lda, int ldb, int ldc) {
    if(m % mTile != 0 || n % nTile != 0 || k % kTile != 0) {
        return -1;
    }
    // Setup TCFGs for compute
    const long aTCFG = 0;
    const long bTCFG = 1;
    const long cTCFG = 2;
    const long aTileTCFG = 3;
    const long bTileTCFG = 4;
    const long cTileTCFG = 5;
    // __tx_tcsr_write is alias for __tx_tcsr_xchg_nmask
    // Mem TCFGs
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(k);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(lda);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(k);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(ldb);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);

    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(ldc);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);

    // NRAM TCFGs
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(mTile);
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
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(mTile); // Float
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM3>(1);

    // Scratchpad for the tile
    void* aTile = 0;
    void* bTile = (void*)((long)aTile + mTile * kTile);
    void* cTile = (void*)((long)bTile + kTile * nTile);

    for(long m_i = 0; m_i < m; m_i += mTile) {
        for(long n_i = 0; n_i < n; n_i += nTile) {
            __tx_fence();
            // Load C
            __tx_async_nram_load<cTCFG, cTileTCFG, 1>((void*)((long)c + (ldc * n_i + m_i) * 2), cTile);
            for(long k_i = 0; k_i < k; k_i += kTile) {
                // load A
                __tx_async_nram_load<aTCFG, aTileTCFG, 0>((void*)((long)a + lda * k_i + m_i), aTile); // src -> dst
                // load B
                __tx_async_nram_load<bTCFG, bTileTCFG, 0>((void*)((long)b + ldb * n_i + k_i), bTile); // src -> dst
                __tx_fence();
                // Compute C = C + A x B
                __tx_matmul_subfloat_e4m3_e4m3_bf16<aTileTCFG, bTileTCFG, cTileTCFG>(aTile, bTile, cTile);
                __tx_fence();
            }
            // Store C
            __tx_async_nram_store<cTileTCFG, cTCFG, 1>(cTile, (void*)((long)c + (ldc * n_i + m_i) * 2));
        }
    }
    __tx_fence();
    return 0;
}
template<long mTile, long nTile, long kTile>
__attribute__((always_inline))
static inline long gmma_s8_s32(void* a, void* b, void* c,
                   int m,   int n,   int k,
                   int lda, int ldb, int ldc) {
    if(m % mTile != 0 || n % nTile != 0 || k % kTile != 0) {
        return -1;
    }
    // Setup TCFGs for compute
    const long aTCFG = 0;
    const long bTCFG = 1;
    const long cTCFG = 2;
    const long aTileTCFG = 3;
    const long bTileTCFG = 4;
    const long cTileTCFG = 5;
    // __tx_tcsr_write is alias for __tx_tcsr_xchg_nmask
    // Mem TCFGs
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(lda);
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

    // NRAM TCFGs
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(mTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(kTile);
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(mTile);
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
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(mTile); // Float
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM3>(1);

    // Scratchpad for the tile
    void* aTile = 0;
    void* bTile = (void*)((long)aTile + mTile * kTile);
    void* cTile = (void*)((long)bTile + kTile * nTile);

    for(long m_i = 0; m_i < m; m_i += mTile) {
        for(long n_i = 0; n_i < n; n_i += nTile) {
            __tx_fence();
            // Load C
            __tx_async_nram_load<cTCFG, cTileTCFG, 2>((void*)((long)c + (ldc * n_i + m_i) * sizeof(int)), cTile);
            for(long k_i = 0; k_i < k; k_i += kTile) {
                // load A
                __tx_async_nram_load<aTCFG, aTileTCFG, 0>((void*)((long)a + lda * k_i + m_i), aTile); // src -> dst
                // load B
                __tx_async_nram_load<bTCFG, bTileTCFG, 0>((void*)((long)b + ldb * n_i + k_i), bTile); // src -> dst
                __tx_fence();
                // Compute C = C + A x B
                __tx_matmul_fix_s8_s8_s32<aTileTCFG, bTileTCFG, cTileTCFG>(aTile, bTile, cTile);
                __tx_fence();
            }
            // Store C
            __tx_async_nram_store<cTileTCFG, cTCFG, 2>(cTile, (void*)((long)c + (ldc * n_i + m_i) * sizeof(int)));
        }
    }
    __tx_fence();
    return 0;
}

extern "C" long gmma_e4m3_bf16_32_32_32(void* a, void* b, void* c,
                   int m,   int n,   int k,
                   int lda, int ldb, int ldc) {
    return gmma_e4m3_bf16<32, 32, 32>(a, b, c, m, n, k, lda, ldb, ldc);
}

extern "C" long gmma_s8_s32_32_32_32(void* a, void* b, void* c,
    int m,   int n,   int k,
    int lda, int ldb, int ldc) {
return gmma_s8_s32<32, 32, 32>(a, b, c, m, n, k, lda, ldb, ldc);
}
