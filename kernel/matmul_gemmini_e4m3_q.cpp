#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

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

#define log2ElementSizeA (0)
#define log2ElementSizeB (0)
#define log2ElementSizeC (2)
#define MaxMTile (32)
#define MaxNTile (32)

static float kernel_buffer[MaxMTile * MaxNTile];
template <long iTile, long jTile, long kTile, long Q_K, long NPUSize>
__attribute__((always_inline)) static inline long
gmma_e4m3q_fp32_BT(void *a, void *b, void *c, int max_i, int max_j, int max_k,
                   int lda, int ldb, int ldc) { /* Setup TCFGs for compute */
  if ((Q_K % kTile) != 0) {
    printf("Not supported: Q_K %% kTile != 0\n");
    return -1;
  }
  if ((max_k % Q_K) != 0) {
    printf("Not supported: max_k %% Q_K != 0\n");
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
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(kTile);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(lda);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);

  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(kTile);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(jTile);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(ldb);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);

  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(jTile);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(ldc);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);

  /* NRAM TCFGs */
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM0>(kTile);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_STRIDE1>(kTile);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<aTileTCFG * 8 + TCSR_DIM3>(1);

  __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM0>(jTile);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE0>(kTile);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM1>(kTile);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<bTileTCFG * 8 + TCSR_DIM3>(1);

  __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM0>(jTile);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM1>(iTile);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_STRIDE1>(jTile);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<cTileTCFG * 8 + TCSR_DIM3>(1);

  /* Scratchpad for the tile */
  const uint64_t acc_addr = (1ULL << 31);
  const uint64_t acc_en_read_full_addr = (1ULL << 29);
  uint64_t aTile = 0;
  uint64_t k_tile_ceil = kTile / NPUSize + !!(kTile % NPUSize);
  uint64_t i_tile_ceil = iTile / NPUSize + !!(iTile % NPUSize);
  uint64_t bTile = aTile + k_tile_ceil * i_tile_ceil * NPUSize;
  uint64_t cPartitialTile = acc_addr;
  // void *cTile = (void *)((long)cPartitialTile + mTile * nTile *
  // sizeof(float));
  float *cF = (float *)c;
  // memset(cF, 0, sizeof(float) * ldc * n);
  // printf("Addr A: 0x%x, B: 0x%x, C: 0x%x\n", a, b, c);
  // printf("M: %d N: %d K: %d\n", m, n, k);

  for (long i = 0; i < max_i; i += iTile) {
    long i_left = max_i - i;
    int real_i_tile = iTile;
    if (i_left < iTile) {
      real_i_tile = i_left;
    }
    updateITile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, true>(
        real_i_tile);
    for (long j = 0; j < max_j; j += jTile) {
      long j_left = max_j - j;
      int real_j_tile = jTile;
      if (j_left < jTile) {
        real_j_tile = j_left;
      }
      updateJTile<aTCFG, bTCFG, cTCFG, aTileTCFG, bTileTCFG, cTileTCFG, true>(
          real_j_tile);
      __tx_fence();
      /* Load C */
      // __tx_async_nram_load<cTCFG, cTileTCFG, log2ElementSizeC>((void
      // *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)), cTile);
      for (long k = 0; k < max_k; k += Q_K) {
        long k_offset = (k / Q_K);
        memset(kernel_buffer, 0, sizeof(float) * real_i_tile * real_j_tile);
        __tx_fence();
        __tx_async_nram_load<cTileTCFG, cTileTCFG, log2ElementSizeC>(
            kernel_buffer, (void *)cPartitialTile);
        // Partital C is computed in each Q_K x Q_K block
        for (long k_i = k; k_i < k + Q_K; k_i += kTile) {
          long real_k_i = k_i + (1 + k_offset) * sizeof(float);
          __tx_async_nram_load<aTCFG, aTileTCFG, log2ElementSizeA>(
              (void *)((long)a +
                       (lda * i + real_k_i) * (1 << log2ElementSizeA)),
              (void *)aTile);
          __tx_async_nram_load<bTCFG, bTileTCFG, log2ElementSizeB>(
              (void *)((long)b +
                       (ldb * j + real_k_i) * (1 << log2ElementSizeB)),
              (void *)bTile);
          __tx_fence(); /* Compute C = C + A x B */
          __tx_matmul_subfloat_e4m3_e4m3_f32<aTileTCFG, bTileTCFG, cTileTCFG>(
              (void *)aTile, (void *)bTile, (void *)cPartitialTile);
          __tx_fence();
        }
        __tx_async_nram_store<cTileTCFG, cTileTCFG, log2ElementSizeC>(
            (void *)(cPartitialTile + acc_en_read_full_addr), kernel_buffer);
        __tx_fence();
        // Scale for the C tile
        for (int scale_i = 0; scale_i < real_i_tile; scale_i += 1) {
          for (int scale_j = 0; scale_j < real_j_tile; scale_j += 1) {
            long real_i = i + scale_i;
            long real_j = j + scale_j;
            float scale_a = *(float *)((long)a + (lda * real_i + k * 1 +
                                                  k_offset * sizeof(float)));
            float scale_b = *(float *)((long)b + (ldb * real_j + k * 1 +
                                                  k_offset * sizeof(float)));
            float scale = scale_a * scale_b;
            // if(real_m == 0 && real_n == 0) {
            //     printf("kernel_buffer[m:%d,n:%d] = %x\n", real_m, real_n,
            //     *(int*)&kernel_buffer[m_ii * realNTile + n_ii]);
            //     printf("cF[m:%d,n:%d] = %x\n", real_m, real_n, *(int*)&cF[ldc
            //     * real_n + real_m]);
            // }
            cF[ldc * real_i + real_j] +=
                kernel_buffer[scale_i * real_j_tile + scale_j] * scale;
            // if(real_m == 0 && real_n == 0) {
            //     printf("kernel_buffer[m:%d,n:%d] = %x\n", real_m, real_n,
            //     *(int*)&kernel_buffer[m_ii * realNTile + n_ii]);
            //     printf("cF[m:%d,n:%d] = %x\n", real_m, real_n, *(int*)&cF[ldc
            //     * real_n + real_m]);
            // }
          }
        }
      }
      __tx_fence();
      // __tx_async_nram_store<cTileTCFG, cTCFG, log2ElementSizeC>(cTile, (void
      // *)((long)c + (ldc * n_i + m_i) * (1 << log2ElementSizeC)));
    }
  }
  __tx_fence();
  return 0;
}

__device__ long gmma_e4m3q_fp32_BT_QK32(void *a, void *b, void *c, int m, int n,
                                        int k, int lda, int ldb, int ldc) {
  return gmma_e4m3q_fp32_BT<16, 16, 16, 32, 4>(a, b, c, m, n, k, lda, ldb, ldc);
}

__device__ long gmma_e4m3q_fp32_BT_QK256(void *a, void *b, void *c, int m,
                                         int n, int k, int lda, int ldb,
                                         int ldc) {
  return gmma_e4m3q_fp32_BT<16, 16, 16, 256, 4>(a, b, c, m, n, k, lda, ldb,
                                                ldc);
}
