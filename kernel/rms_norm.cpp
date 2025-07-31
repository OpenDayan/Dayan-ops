#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <cmath>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long oneTCFG = 3;

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

static inline void configoneTCSR(void) {
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM0>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM1>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE1>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE2>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE3>(1);
}

__device__ long rms_norm_f32(float *x, float *y, int b, int h, int n, int m, float epsilon) {
  float m0 = 1.0f / (float)m;

  float* XTile  = (float*)NULL;
  float* YTile  = XTile  + m;
  float* mid    = YTile  + m;
  float* ave    = mid    + m;
  float* norm   = ave    + 1;
  float* mram   = norm   + 1;

  configoneTCSR();
  configTCSRs(m, 1);

  for (int i = 0; i < b; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < n; ++k) {
        const int64_t head_offset = (i * h * n + j * n + k) * m;

        const float* x_ptr = &x[head_offset];
        float* y_ptr = &y[head_offset];

        __tx_async_nram_load<aTCSR, aTCSR, 2>((float*)x_ptr, XTile);
        __tx_async_nram_load<oneTCFG, oneTCFG, 2>(&m0, mram);
        __tx_fence();

        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0>(XTile, XTile, mid);
        __tx_fence();

        __tx_fp_reduce_add_fp32<aTCSR, oneTCFG, 0>(mid, norm);                      //sum
        __tx_fence();

        __tx_fp_mul_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(norm, mram, ave);
        __tx_fence();
        __tx_fp_sub_fp32<oneTCFG, oneTCFG, oneTCFG, 0b0000>(ave, &epsilon, mram);   //ave
        __tx_fence();


        __tx_sfu_rsqrt_fp32<oneTCFG, oneTCFG>(mram, norm);
        __tx_fence();
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile, norm, YTile);
        __tx_fence();
        
        __tx_async_nram_store<cTCSR, cTCSR, 2>(YTile, (float*)y_ptr);
        __tx_fence();
      }
    }
  }
  return 0;
}