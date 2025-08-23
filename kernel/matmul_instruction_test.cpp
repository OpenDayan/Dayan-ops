#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

static const long aTCFG = 0;
static const long bTCFG = 1;
static const long cTCFG = 2;

static const uint64_t uram0 = 0x1000000;
static const uint64_t uram1 = 0x2000000;
static const uint64_t uram2 = 0x3000000;

static inline void config_row_major_tcsr0(int m, int n) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE3>(m * n);
}

static inline void config_row_major_tcsr1(int m, int n) {
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE3>(m * n);
}

static inline void config_row_major_tcsr2(int m, int n) {
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE3>(m * n);
}


static inline void config_col_major_tcsr0(int m, int n) {
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<aTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<aTCFG * 8 + TCSR_STRIDE3>(m * n);
}

static inline void config_col_major_tcsr1(int m, int n) {
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<bTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<bTCFG * 8 + TCSR_STRIDE3>(m * n);
}

static inline void config_col_major_tcsr2(int m, int n) {
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE2>(m * n);
  __tx_tcsr_write<cTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<cTCFG * 8 + TCSR_STRIDE3>(m * n);
}

// input0 and input1 both row major
__device__ void matmul_fp32_fp32_fp32_row_row(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input0, (void *)uram0);

  config_row_major_tcsr0(k, n);
  config_row_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_float_f32_f32_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                         (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

// input0 is row major and input1 is col major
__device__ void matmul_fp32_fp32_fp32_row_col(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input0, (void *)uram0);

  config_col_major_tcsr0(k, n);
  config_col_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_row_major_tcsr0(m, k);
  config_col_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_float_f32_f32_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                         (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

// input0 is col major and input1 is row major
__device__ void matmul_fp32_fp32_fp32_col_row(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_col_major_tcsr0(m, k);
  config_col_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input0, (void *)uram0);

  config_row_major_tcsr0(k, n);
  config_row_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_col_major_tcsr0(m, k);
  config_row_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_float_f32_f32_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                         (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

// input0 is col major and input1 is col major
__device__ void matmul_fp32_fp32_fp32_col_col(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_col_major_tcsr0(m, k);
  config_col_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input0, (void *)uram0);

  config_col_major_tcsr0(k, n);
  config_col_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_col_major_tcsr0(m, k);
  config_col_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_float_f32_f32_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                         (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

// input0 and input1 both row major
__device__ void matmul_fp16_fp16_fp32_row_row(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_row_major_tcsr0(m, k);
  __tx_async_uram_load<aTCFG, aTCFG, 2>(input0, (void *)uram0);

  config_row_major_tcsr0(k, n);
  __tx_async_uram_load<aTCFG, aTCFG, 2>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_float_fp16_fp16_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                         (void *)uram2);

  config_row_major_tcsr0(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, aTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

__device__ void matmul_bf16_bf16_fp32_row_row(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_row_major_tcsr0(m, k);
  __tx_async_uram_load<aTCFG, aTCFG, 2>(input0, (void *)uram0);

  config_row_major_tcsr0(k, n);
  __tx_async_uram_load<aTCFG, aTCFG, 2>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_float_bf16_bf16_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                         (void *)uram2);

  config_row_major_tcsr0(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, aTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}


__device__ void matmul_e4m3_e4m3_fp32_row_row(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input0, (void *)uram0);

  config_row_major_tcsr0(k, n);
  config_row_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_subfloat_e4m3_e4m3_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                              (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

__device__ void matmul_e4m3_e4m3_fp32_row_col(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_row_major_tcsr0(m, k);
  config_row_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input0, (void *)uram0);

  config_col_major_tcsr0(k, n);
  config_col_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_row_major_tcsr0(m, k);
  config_col_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_subfloat_e4m3_e4m3_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                              (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

__device__ void matmul_e4m3_e4m3_fp32_col_row(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_col_major_tcsr0(m, k);
  config_col_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input0, (void *)uram0);

  config_row_major_tcsr0(k, n);
  config_row_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_col_major_tcsr0(m, k);
  config_row_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_subfloat_e4m3_e4m3_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                              (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

__device__ void matmul_e4m3_e4m3_fp32_col_col(void *input0, void *input1, void *dst,
                                      int m, int n, int k) {

  config_col_major_tcsr0(m, k);
  config_col_major_tcsr2(m, k);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input0, (void *)uram0);

  config_col_major_tcsr0(k, n);
  config_col_major_tcsr2(k, n);
  __tx_async_uram_load<aTCFG, cTCFG, 0>(input1, (void *)uram1);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_async_uram_load<aTCFG, cTCFG, 2>(dst, (void *)uram2);

  config_col_major_tcsr0(m, k);
  config_col_major_tcsr1(k, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_matmul_subfloat_e4m3_e4m3_f32<0, 1, 2>((void *)uram0, (void *)uram1,
                                              (void *)uram2);

  config_row_major_tcsr0(m, n);
  config_row_major_tcsr2(m, n);
  __tx_fence();
  __tx_async_uram_store<aTCFG, cTCFG, 2>((void *)uram2, dst);
  __tx_fence();
}

