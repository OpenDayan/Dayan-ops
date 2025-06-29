#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCFG = 0;
const long bTCFG = 1;
const long cTCFG = 2;

static inline void configTCSRs(int m, int n) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE3>(m * n);

  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE3>(m * n);

  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE3>(m * n);
}
__device__ void tx_int_add_s32(void *input0, void *input1, void *dst, int m,
                               int n) {
  configTCSRs(m, n);
  __tx_fence();
  // TODO: copy inputs to nram
  __tx_int_add_s32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}

__device__ void tx_int_sub_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  // TODO: copy inputs to nram
  __tx_int_sub_s32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}

__device__ void tx_int_mul_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  // TODO: copy inputs to nram
  __tx_int_mul_s32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}

__device__ void tx_fp_add_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  // TODO: copy inputs to nram
  __tx_fp_add_fp32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}

__device__ void tx_fp_sub_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  // TODO: copy inputs to nram
  __tx_fp_sub_fp32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}

__device__ void tx_fp_mul_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  // TODO: copy inputs to nram
  __tx_fp_mul_fp32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}