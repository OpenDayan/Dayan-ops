#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

template <long aTCFG, long bTCFG, long cTCFG, long aTileTCFG, long bTileTCFG,
          long cTileTCFG, bool B_T>
static inline void updateITile(long iTile) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM1>(iTile);
}

template <long aTCFG, long bTCFG, long cTCFG, long aTileTCFG, long bTileTCFG,
          long cTileTCFG, bool B_T>
static inline void updateJTile(long jTile) {
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(jTile);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM0>(jTile);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM0>(jTile);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(jTile);
  if (B_T) {
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(jTile);
  } else {
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(jTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE1>(jTile);
  }
}

template <long aTCFG, long bTCFG, long cTCFG, long aTileTCFG, long bTileTCFG,
          long cTileTCFG, bool B_T>
static inline void updateKTile(long kTile) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(kTile);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(kTile);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(kTile);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM1>(kTile);
  if (B_T) {
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(kTile);
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE0>(kTile);
  } else {
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(kTile);
  }
}

// Note: A, B, C are column-major matrices
// B **CAN BE** Transposed
// A[i][k]
// B[k][j]; B[j][k]
// C[i][j]
// C = C + A x B
// C, A: row-major, B: row-major/col-major
#define GEN_MATRIX_KERNEL(name, log2ElementSizeA, log2ElementSizeB,            \
                          log2ElementSizeC, matmulFunc)                        \
  template <long iTile, long jTile, long kTile, bool B_T, long NPUSize>        \
  __attribute__((always_inline)) static inline long name(                      \
      void *a, void *b, void *c, int max_i, int max_j, int max_k, int lda,     \
      int ldb, int ldc) {                                                      \
    /* Setup TCFGs for compute */                                              \
    const long aTCFG = 0;                                                      \
    const long bTCFG = 1;                                                      \
    const long cTCFG = 2;                                                      \
    const long aTileTCFG = 3;                                                  \
    const long bTileTCFG = 4;                                                  \
    const long cTileTCFG = 5;                                                  \
    /* __tx_tcsr_write is alias for __tx_tcsr_xchg_nmask */                    \
    /* Mem TCFGs */                                                            \
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(kTile);                             \
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);                              \
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(iTile);                             \
    __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(lda);                            \
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);                                 \
    __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);                                 \
                                                                               \
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(B_T ? kTile : jTile);               \
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);                              \
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(B_T ? jTile : kTile);               \
    __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(ldb);                            \
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);                                 \
    __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);                                 \
                                                                               \
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(jTile);                             \
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);                              \
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(iTile);                             \
    __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(ldc);                            \
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);                                 \
    __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);                                 \
                                                                               \
    /* URAM TCFGs */                                                           \
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(kTile);                         \
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(1);                          \
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(iTile);                         \
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(kTile);                      \
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM2>(1);                             \
    __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM3>(1);                             \
                                                                               \
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM0>(jTile);                         \
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE0>(B_T ? kTile : 1);            \
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM1>(kTile);                         \
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE1>(B_T ? 1 : jTile);            \
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM2>(1);                             \
    __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM3>(1);                             \
                                                                               \
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM0>(jTile);                         \
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE0>(1);                          \
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM1>(iTile);                         \
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(jTile);                      \
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM2>(1);                             \
    __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM3>(1);                             \
                                                                               \
    /* Scratchpad for the tile */                                              \
    const uint64_t acc_addr = (1ULL << 31);                                    \
    const uint64_t acc_en_read_full_addr = (1ULL << 29);                       \
    uint64_t aTile = 0;                                                        \
    uint64_t k_tile_ceil = kTile / NPUSize + !!(kTile % NPUSize);              \
    uint64_t i_tile_ceil = iTile / NPUSize + !!(iTile % NPUSize);              \
    uint64_t bTile = aTile + k_tile_ceil * i_tile_ceil * NPUSize;              \
    uint64_t cTile = acc_addr;                                                 \
                                                                               \
    for (size_t i = 0; i < max_i; i += iTile) {                                \
      size_t i_left = max_i - i;                                               \
      if (i_left < iTile) {                                                    \
        updateITile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG,      \
                    B_T>(i_left);                                              \
      }                                                                        \
      updateJTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, B_T>(  \
          jTile);                                                              \
      for (size_t j = 0; j < max_j; j += jTile) {                              \
        size_t j_left = max_j - j;                                             \
        if (j_left < jTile) {                                                  \
          updateJTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG,    \
                      B_T>(j_left);                                            \
        }                                                                      \
        __tx_fence();                                                          \
        /* Load C */                                                           \
        __tx_async_uram_load<cTCFG, cTileTCFG, log2ElementSizeC>(              \
            (void *)((uint64_t)c + (ldc * i + j) * (1 << log2ElementSizeC)), \
            (void *)cTile);                                                    \
        updateKTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG,      \
                    B_T>(kTile);                                               \
        for (long k = 0; k < max_k; k += kTile) {                              \
          long k_left = max_k - k;                                             \
          if (k_left < kTile) {                                                \
            updateKTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG,  \
                        B_T>(k_left);                                          \
          }                                                                    \
          /* load A */                                                         \
          __tx_async_uram_load<aTCFG, aTileTCFG, log2ElementSizeA>(            \
              (void *)((uint64_t)a +                                           \
                       (lda * i + k) * (1 << log2ElementSizeA)),             \
              (void *)aTile);                                                  \
          /* load B */                                                         \
          __tx_async_uram_load<bTCFG, bTileTCFG, log2ElementSizeB>(            \
              (void *)(B_T ? ((uint64_t)b +                                    \
                              (ldb * j + k) * (1 << log2ElementSizeB))       \
                           : ((uint64_t)b +                                    \
                              (ldb * k + j) * (1 << log2ElementSizeB))),     \
              (void *)bTile);                                                  \
          __tx_fence();                                                        \
          /* Compute C = C + A x B */                                          \
          matmulFunc<aTileTCFG, bTileTCFG, cTileTCFG>(                         \
              (void *)aTile, (void *)bTile, (void *)cTile);                    \
          __tx_fence();                                                        \
        }                                                                      \
        /* Store C */                                                          \
        __tx_async_uram_store<cTileTCFG, cTCFG, log2ElementSizeC>(             \
            (void *)(cTile + acc_en_read_full_addr),                           \
            (void *)((uint64_t)c +                                             \
                     (ldc * i + j) * (1 << log2ElementSizeC)));              \
      }                                                                        \
    }                                                                          \
    __tx_fence();                                                              \
    return 0;                                                                  \
  }

// compute C := A * B + C
// C, B: col-major, A: row-major
// C^T := B^T * A^T + C^T
// let C_1 = C^T, A_1 = B^T, B_1 = A^T
// C_1 := A_1 * B_1 + C_1
// C_1, A_1: row-major, B_1: col-major

GEN_MATRIX_KERNEL(gmma_e4m3_e4m3_f32, 0, 0, 2,
                  __tx_matmul_subfloat_e4m3_e4m3_f32)
GEN_MATRIX_KERNEL(gmma_e4m3_e4m3_bf16, 0, 0, 1,
                  __tx_matmul_subfloat_e4m3_e4m3_bf16)
GEN_MATRIX_KERNEL(gmma_s8_s8_s32, 0, 0, 2, __tx_matmul_fix_s8_s8_s32)

__device__ long gmma_gemmini_e4m3_f32_16_16_16(void *a, void *b, void *c, int m, int n,
                                       int k, int lda, int ldb, int ldc) {
  return gmma_e4m3_e4m3_f32<16, 16, 16, false, 4>(a, b, c, m, n, k, lda, ldb,
                                                  ldc);
}

__device__ long gmma_gemmini_e4m3_bf16_16_16_16(void *a, void *b, void *c, int m, int n,
                                        int k, int lda, int ldb, int ldc) {
  return gmma_e4m3_e4m3_bf16<16, 16, 16, false, 4>(a, b, c, m, n, k, lda, ldb,
                                                   ldc);
}

__device__ long gmma_gemmini_s8_s32_16_16_16(void *a, void *b, void *c, int m, int n,
                                     int k, int lda, int ldb, int ldc) {
  return gmma_s8_s8_s32<16, 16, 16, false, 16>(a, b, c, m, n, k, lda, ldb, ldc);
}

__device__ long gmma_e4m3_bf16_16_16_16_BT(void *a, void *b, void *c, int m,
                                           int n, int k, int lda, int ldb,
                                           int ldc) {
  return gmma_e4m3_e4m3_bf16<16, 16, 16, true, 4>(a, b, c, m, n, k, lda, ldb,
                                                  ldc);
}

__device__ long gmma_e4m3_f32_16_16_16_BT(void *a, void *b, void *c, int m,
                                          int n, int k, int lda, int ldb,
                                          int ldc) {
  return gmma_e4m3_e4m3_f32<16, 16, 16, true, 4>(a, b, c, m, n, k, lda, ldb,
                                                 ldc);
}

__device__ long gmma_s8_s32_16_16_16_BT(void *a, void *b, void *c, int m, int n,
                                        int k, int lda, int ldb, int ldc) {
  return gmma_s8_s8_s32<16, 16, 16, true, 16>(a, b, c, m, n, k, lda, ldb, ldc);
}
