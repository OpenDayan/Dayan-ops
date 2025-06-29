#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCFG = 0;
const long bTCFG = 1;
const long cTCFG = 2;

__device__ void matmul_fp32_fp32_fp32(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(k);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(k);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(n);

  __tx_fence();
  // TODO: copy inputs to nram

  __tx_matmul_float_f32_f32_f32<0, 1, 2>((void *)input0, (void *)input1,
                                         (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}

__device__ void matmul_e4m3_e4m3_fp32(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(k);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(k);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(n);
  __tx_fence();
  // TODO: copy inputs to nram

  __tx_matmul_subfloat_e4m3_e4m3_f32<0, 1, 2>((void *)input0, (void *)input1,
                                              (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}
