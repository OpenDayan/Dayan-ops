#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCFG = 0;
const long bTCFG = 1;
const long cTCFG = 2;

const uint64_t nram0 = 0x1000000;
const uint64_t nram1 = 0x2000000;
const uint64_t nram2 = 0x3000000;

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
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_add_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sub_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_sub_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_mul_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_mul_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_srl_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_srl_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sra_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_sra_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sll_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_sll_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_add_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_fp_add_fp32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_sub_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {
  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_fp_sub_fp32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_mul_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_fp_mul_fp32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_max_fp32(void *input0,  void *dst, int m,
                               int n, const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis  == 1)  // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
     __tx_fp_reduce_max_fp32<aTCFG, cTCFG, 0>((void *)nram0, (void *)nram2);
  if (axis  == 1)
     __tx_fp_reduce_max_fp32<aTCFG, cTCFG, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis  == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_min_fp32(void *input0,  void *dst, int m,
                               int n, const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis  == 1)  // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
     __tx_fp_reduce_min_fp32<aTCFG, cTCFG, 0>((void *)nram0, (void *)nram2);
  if (axis  == 1)
     __tx_fp_reduce_min_fp32<aTCFG, cTCFG, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis  == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_add_fp32(void *input0,  void *dst, int m,
                               int n, const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCFG, cTCFG, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis  == 1)  // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCFG, cTCFG, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
     __tx_fp_reduce_add_fp32<aTCFG, cTCFG, 0>((void *)nram0, (void *)nram2);
  if (axis  == 1)
     __tx_fp_reduce_add_fp32<aTCFG, cTCFG, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis  == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCFG, cTCFG, 2>((void *)nram2, dst);
  __tx_fence();
}