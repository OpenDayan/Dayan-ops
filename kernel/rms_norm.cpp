#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <cmath>

__device__ long rms_norm_f32(void *x, void *y, void *m0, int b, int h, int n, int m, float epsilon) {
  const long xTCFG = 0;
  const long yTCFG = 1;
  const long normTCFG = 2;
  const long oneTCFG = 3;

  const uint64_t stride0 = 1;
  const uint64_t stride1 = stride0 * m;
  const uint64_t stride2 = stride0 * m * n;
  const uint64_t stride3 = stride0 * m * n * h;

  __tx_tcsr_write<xTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<xTCFG * 8 + TCSR_STRIDE0>(stride0);
  __tx_tcsr_write<xTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<xTCFG * 8 + TCSR_STRIDE1>(stride1);
  __tx_tcsr_write<xTCFG * 8 + TCSR_DIM2>(h);
  __tx_tcsr_write<xTCFG * 8 + TCSR_STRIDE2>(stride2);
  __tx_tcsr_write<xTCFG * 8 + TCSR_DIM3>(b);
  __tx_tcsr_write<xTCFG * 8 + TCSR_STRIDE3>(stride3);

  __tx_tcsr_write<yTCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<yTCFG * 8 + TCSR_STRIDE0>(stride0);
  __tx_tcsr_write<yTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<yTCFG * 8 + TCSR_STRIDE1>(stride1);
  __tx_tcsr_write<yTCFG * 8 + TCSR_DIM2>(h);
  __tx_tcsr_write<yTCFG * 8 + TCSR_STRIDE2>(stride2);
  __tx_tcsr_write<yTCFG * 8 + TCSR_DIM3>(b);
  __tx_tcsr_write<yTCFG * 8 + TCSR_STRIDE3>(stride3);

  __tx_tcsr_write<normTCFG * 8 + TCSR_DIM0>(1);
  __tx_tcsr_write<normTCFG * 8 + TCSR_STRIDE0>(stride0);
  __tx_tcsr_write<normTCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<normTCFG * 8 + TCSR_STRIDE1>(stride1 / m);
  __tx_tcsr_write<normTCFG * 8 + TCSR_DIM2>(h);
  __tx_tcsr_write<normTCFG * 8 + TCSR_STRIDE2>(stride2 / m);
  __tx_tcsr_write<normTCFG * 8 + TCSR_DIM3>(b);
  __tx_tcsr_write<normTCFG * 8 + TCSR_STRIDE3>(stride3 / m);

  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM0>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM1>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE2>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE3>(1);

  float* XTile          = (float*)NULL;
  float* kernel_buffer  = XTile          + b * h * n * m;
  float* mid            = kernel_buffer  + b * h * n * m;
  float* Average        = mid            + b * h * n * m;
  float* norm           = Average        + b * h * n;
  float* m0ram          = norm           + b * h * n;

  __tx_async_nram_load<xTCFG, xTCFG, 2>(
    x, XTile);
  __tx_async_nram_load<oneTCFG, oneTCFG, 2>(
    m0, m0ram);
  __tx_fence();
  __tx_fp_mul_fp32<0, 1, 2, 0>(
    XTile, XTile, mid);
  __tx_fence();
  __tx_fp_reduce_add_fp32<xTCFG, normTCFG, 0>(        //sum
    mid, norm);
  __tx_fence();

  __tx_fp_mul_fp32<normTCFG, oneTCFG, normTCFG, 0b1110>(   //ave
    norm, m0ram, Average);
  __tx_fence();

  __tx_sfu_rsqrt_fp32<normTCFG, normTCFG>(
    Average, m0ram);
  __tx_fence();
  __tx_fp_mul_fp32<xTCFG, normTCFG, yTCFG, 0b0001>(
    XTile, m0ram, kernel_buffer);
  __tx_fence();
  
  __tx_async_nram_store<yTCFG, yTCFG, 2>(
    kernel_buffer, y);
  __tx_fence();
  return 0;
}