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


__device__ long silu_f32(float *x, float *y, int b, int h, int n, int m) {
  const int aTile = 256;

  float minus_one = -1.0f;

  float* XTile     = (float*)NULL;
  float* YTile     = XTile  + aTile;
  float* mid1      = YTile  + aTile;
  float* mid2      = mid1   + aTile;
  float* minusone  = mid2   + aTile;

  configoneTCSR();

  for (int i = 0; i < b; ++i) {
    for (int j = 0; j < h; ++j) {
      for (int k = 0; k < n; ++k) {
        for (int l = 0; l < m; l += aTile) {
          int m_i = aTile;
          if(l + aTile > m) {
            m_i = m - l;
          }
          configTCSRs(m_i, 1);
          
          const int64_t head_offset = (i * h * n + j * n + k) * m + l;

          const float* x_ptr = &x[head_offset];
          float* y_ptr = &y[head_offset];

          __tx_async_uram_load<aTCSR, aTCSR, 2>((float*)x_ptr, XTile);
          __tx_async_uram_load<oneTCFG, oneTCFG, 2>(&minus_one, minusone);
          __tx_fence();

          __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile, minusone, mid1);    //mid1 = -x
          __tx_fence();
          __tx_sfu_exp_fp32<aTCSR, aTCSR>(mid1, mid2);                               //mid2 = e^(-x)
          __tx_fence();
          __tx_fp_sub_fp32<aTCSR, oneTCFG, aTCSR, 0b0001>(mid2, minusone, mid1);     //mid1 = 1+e^(-x)
          __tx_fence();
          __tx_sfu_recip_fp32<aTCSR, aTCSR>(mid1, mid2);                             //mid2 = 1/(1+e^(-x))^(1/2)
          __tx_fence();
          __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(XTile, mid2, YTile);         //y = x/(1+e^(-x))^(1/2)
          __tx_fence();

          __tx_async_uram_store<cTCSR, cTCSR, 2>(YTile, (float*)y_ptr);
          __tx_fence();
        }
      }
    }
  }
  return 0;
}