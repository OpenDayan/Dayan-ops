#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <cmath>

#define MaxMTile (32)
#define MaxNTile (32)
#define MaxhTile (32)
#define MaxbTile (32)

static float kernel_buffer[0x1000000];
__device__ long silu_f32(void *x, void *y, void* minus_one, int b, int h, int n, int m) {
  const long xTCFG = 0;
  const long yTCFG = 1;
  const long mid1TCFG = 2;
  const long mid2TCFG = 3;
  const long oneTCFG = 4;
  
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

  __tx_tcsr_write<mid1TCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_STRIDE0>(stride0);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_STRIDE1>(stride1);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_DIM2>(h);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_STRIDE2>(stride2);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_DIM3>(b);
  __tx_tcsr_write<mid1TCFG * 8 + TCSR_STRIDE3>(stride3);

  __tx_tcsr_write<mid2TCFG * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_STRIDE0>(stride0);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_DIM1>(n);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_STRIDE1>(stride1);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_DIM2>(h);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_STRIDE2>(stride2);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_DIM3>(b);
  __tx_tcsr_write<mid2TCFG * 8 + TCSR_STRIDE3>(stride3);

  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM0>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM1>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE2>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE3>(1);


  float* XTile     = (float*)NULL;
  float* mid1      = XTile     + b * h * n * m;
  float* mid2      = mid1      + b * h * n * m;
  float* minusone  = mid2      + b * h * n * m;


  __tx_async_nram_load<xTCFG, xTCFG, 2>(
    x, XTile);
  __tx_async_nram_load<oneTCFG, oneTCFG, 2>(
    minus_one, minusone);
  __tx_fence();

  __tx_fp_mul_fp32<xTCFG, oneTCFG, mid1TCFG, 0b1111>(               //mid1 = -x
    XTile, minusone, mid1);
  __tx_fence();
  __tx_sfu_exp_fp32<mid1TCFG, mid2TCFG>(                            //mid2 = e^(-x)
    mid1, mid2);
  __tx_fence();
  __tx_fp_sub_fp32<mid2TCFG, oneTCFG, mid1TCFG, 0b1111>(               //mid1 = 1+e^(-x)
    mid2, minusone, mid1);
  __tx_fence();
  __tx_sfu_recip_fp32<mid1TCFG, mid2TCFG>(                          //mid2 = 1/(1+e^(-x))^(1/2)
    mid1, mid2);
  __tx_fence();
  __tx_fp_mul_fp32<xTCFG, mid2TCFG, yTCFG, 0>(              //y = x/(1+e^(-x))
    XTile, mid2, kernel_buffer);
  __tx_fence();

  __tx_async_nram_store<yTCFG, yTCFG, 2>(
    kernel_buffer, y);
  __tx_fence();
  return 0;
}