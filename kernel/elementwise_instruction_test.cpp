#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const uint64_t nram0 = 0x1000000;
const uint64_t nram1 = 0x2000000;
const uint64_t nram2 = 0x3000000;

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

static inline void config_dims_and_strides(TCFG *tcsr) {
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(tcsr[0].dim0);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(tcsr[0].stride0);
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(tcsr[0].dim1);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(tcsr[0].stride1);
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(tcsr[0].dim2);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(tcsr[0].stride2);
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(tcsr[0].dim3);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(tcsr[0].stride3);

  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(tcsr[1].dim0);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(tcsr[1].stride0);
  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(tcsr[1].dim1);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(tcsr[1].stride1);
  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(tcsr[1].dim2);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(tcsr[1].stride2);
  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(tcsr[1].dim3);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(tcsr[1].stride3);


  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(tcsr[2].dim0);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(tcsr[2].stride0);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(tcsr[2].dim1);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(tcsr[2].stride1);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(tcsr[2].dim2);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(tcsr[2].stride2);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(tcsr[2].dim3);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(tcsr[2].stride3);
}

__device__ void tx_int_add_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {
  config_dims_and_strides(tcsr);
  // __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  // __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  // __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  // __tx_fence();

  // __tx_int_add_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  // __tx_fence();

  // __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  // __tx_fence();
}

__device__ void tx_int_sub_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_fence();
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_sub_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_mul_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_mul_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_srl_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_srl_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sra_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_sra_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sll_s32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_int_sll_s32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_lutop_s32(void *input0, void *input1, void *dst, int m,
                                 int n, uint8_t lutop) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();
  lutop = lutop & 0b1111;

  switch (lutop) {
  case 0:
    __tx_int_lutop_s32<0, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 1:
    __tx_int_lutop_s32<1, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 2:
    __tx_int_lutop_s32<2, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 3:
    __tx_int_lutop_s32<3, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 4:
    __tx_int_lutop_s32<4, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 5:
    __tx_int_lutop_s32<5, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 6:
    __tx_int_lutop_s32<6, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 7:
    __tx_int_lutop_s32<7, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 8:
    __tx_int_lutop_s32<8, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 9:
    __tx_int_lutop_s32<9, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                      (void *)nram2);
    break;
  case 10:
    __tx_int_lutop_s32<10, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                       (void *)nram2);
    break;
  case 11:
    __tx_int_lutop_s32<11, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                       (void *)nram2);
    break;
  case 12:
    __tx_int_lutop_s32<12, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                       (void *)nram2);
    break;
  case 13:
    __tx_int_lutop_s32<13, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                       (void *)nram2);
    break;
  case 14:
    __tx_int_lutop_s32<14, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                       (void *)nram2);
    break;
  case 15:
    __tx_int_lutop_s32<15, 0, 1, 2, 0>((void *)nram0, (void *)nram1,
                                       (void *)nram2);
    break;
  }
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_add_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_fp_add_fp32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_sub_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {
  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_fp_sub_fp32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_mul_fp32(void *input0, void *input1, void *dst, int m,
                               int n) {

  configTCSRs(m, n);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  __tx_fp_mul_fp32<0, 1, 2, 0>((void *)nram0, (void *)nram1, (void *)nram2);
  __tx_fence();

  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_max_fp32(void *input0, void *dst, int m, int n,
                                      const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis == 1) // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
    __tx_fp_reduce_max_fp32<aTCSR, cTCSR, 0>((void *)nram0, (void *)nram2);
  if (axis == 1)
    __tx_fp_reduce_max_fp32<aTCSR, cTCSR, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_min_fp32(void *input0, void *dst, int m, int n,
                                      const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis == 1) // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
    __tx_fp_reduce_min_fp32<aTCSR, cTCSR, 0>((void *)nram0, (void *)nram2);
  if (axis == 1)
    __tx_fp_reduce_min_fp32<aTCSR, cTCSR, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_add_fp32(void *input0, void *dst, int m, int n,
                                      const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis == 1) // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
    __tx_fp_reduce_add_fp32<aTCSR, cTCSR, 0>((void *)nram0, (void *)nram2);
  if (axis == 1)
    __tx_fp_reduce_add_fp32<aTCSR, cTCSR, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}