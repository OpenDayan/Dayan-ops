#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long oneTCSR = 3;

const uint64_t nram0 = 0x0100000;
const uint64_t nram1 = 0x0200000;

template<int TCSR_ID>
static inline void config_tcsr_vector(int m) {
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM0>(m);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE0>(1);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM1>(1);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE1>(m);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM2>(1);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE2>(m);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM3>(1);
  __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE3>(m);
}

__device__ long softmax_f32_f32_f32(float* a, float* b, float* res, uint32_t scale_,
              int bs, int h, int n, int m) {
  float scale = u32_as_float(scale_);
  float* sp = (float*)NULL;
  float* scale_ram = sp + 1;
  float* slope_ram = sp + 2;
  float* max_ram = sp + 3;
  float* sum_ram = sp + 4;

  config_tcsr_vector<aTCSR>(m);
  config_tcsr_vector<bTCSR>(m);
  config_tcsr_vector<cTCSR>(m);
  config_tcsr_vector<oneTCSR>(1);

  for (int i = 0; i < bs; ++i) { // batch
    for (int j = 0; j < (h * n); ++j) { // attn-heads * seq_len
      const uint32_t hidx = (j / n); // current head idx
      // const float slope = (max_bias > 0.0f) ? h < h_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - h_log2) + 1) : 1.0f;
      float slope = 1.0f; // only support max_bias == 0.0f in qwen2

      float* a_ptr = &a[i * h * n * m + j * m];
      float* b_ptr = &b[i * h * n * m + (j % n) * m];
      float* res_ptr = &res[i * h * n * m + j * m];

      float sum = 0.0f;

      // load stage
      __tx_async_nram_load<aTCSR, aTCSR, 2>(a_ptr, (void*)nram0);
      __tx_async_nram_load<bTCSR, bTCSR, 2>(b_ptr, (void*)nram1);
      __tx_async_nram_load<oneTCSR, oneTCSR, 2>(&scale, scale_ram);
      __tx_async_nram_load<oneTCSR, oneTCSR, 2>(&slope, slope_ram);
      __tx_fence();

      // compute stage
      __tx_fp_mul_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>((void*)nram0, scale_ram, (void*)nram0); // scale: a[i][j][k][:] *= scale
      __tx_fp_mul_fp32<bTCSR, oneTCSR, cTCSR, 0b0001>((void*)nram1, slope_ram, (void*)nram1); // mask:  temp = slope * b[i][0][k][:]
      __tx_fence();
      __tx_fp_add_fp32<aTCSR, cTCSR, aTCSR, 0b0000>((void*)nram0, (void*)nram1, (void*)nram0); // mask: a[i][j][k][:] += temp
      __tx_fence();

      __tx_fp_reduce_max_fp32<aTCSR, oneTCSR, 1>((void*)nram0, max_ram); // maxval = max(a[i][j][k][:])
      __tx_fence();

      __tx_fp_sub_fp32<aTCSR, oneTCSR, aTCSR, 0b0001>((void*)nram0, (void*)max_ram, (void*)nram0);  // a[i][j][k][:] = a[i][j][k][:] - max
      __tx_fence();
      __tx_sfu_exp_fp32<aTCSR, bTCSR>((void*)nram0, (void*)nram1);  // exp_temp = exp(a[i][j][k][:])
      __tx_fence();

      __tx_fp_reduce_add_fp32<bTCSR, oneTCSR, 0>((void*)nram1, (void*)sum_ram); // sum = sum of exp_temp
      __tx_fence();
      __tx_async_nram_store<oneTCSR, oneTCSR, 2>((void*)sum_ram, &sum);
      __tx_fence();

      sum = 1.0f / sum;

      __tx_async_nram_load<oneTCSR, oneTCSR, 2>(&sum, sum_ram);
      __tx_fence();
      __tx_fp_mul_fp32<bTCSR, oneTCSR, cTCSR, 0b0001>((void*)nram1, sum_ram, (void*)nram1); // exp_temp *= 1/sum
      __tx_fence();

      // store stage
      __tx_async_nram_store<bTCSR, bTCSR, 2>((void*)nram1, res_ptr);  // res[i][j][k][:] = exp_temp
      __tx_fence();
    }
  }
  return 0;
}