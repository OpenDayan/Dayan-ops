#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long dTCSR = 3;
const long oneTCSR = 4;

const uint64_t uram0 = 0x0100000;
const uint64_t uram1 = 0x0200000;
const uint64_t uram2 = 0x0300000;
const uint64_t uram3 = 0x0400000;
const uint64_t uram4 = 0x0500000;
const uint64_t uram5 = 0x0600000;
const uint64_t uram6 = 0x0700000;
const uint64_t uram7 = 0x0800000;

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

static inline void rope_yarn(
        float* cos_cache,
        float* sin_cache,
        float theta_base, float theta_scale,
        float freq_scale,
        int m,
        float* corr_dims = NULL, // useless in qwen2
        float* freq_factor = NULL, // freq_factor is always NULL in qwen2
        float ext_factor = 0.0f, // // ext_factor is always 0.0f in qwen2
        float mscale = 1.0f, // mscale == attn_factor is always 1.0f in qwen2
        float sin_sign = 1.0f // sin_sign is always 1.0f in forward computing
) {
  float* sp = (float*)NULL;
  float* rama = sp + 1;
  float* ramb = sp + 2;
  float* ramc = sp + 3;
  float theta = theta_base;
  for (int i = 0; i < m / 2; i++) {
    const float ff = 1.0f; // freq_factor ? freq_factor[i / 2] : 1.0f;
    float theta_extrap = theta / ff;
    float theta_interp = freq_scale * theta_extrap;
    float cos, sin;
    __tx_async_uram_load<oneTCSR, oneTCSR, 2>(&theta_interp, rama);
    __tx_fence();
    __tx_sfu_cos_fp32<oneTCSR, oneTCSR>(rama, ramb);
    __tx_fence();
    __tx_sfu_sin_fp32<oneTCSR, oneTCSR>(rama, ramc);
    __tx_fence();
    __tx_async_uram_store<oneTCSR, oneTCSR, 2>(ramb, &cos);
    __tx_async_uram_store<oneTCSR, oneTCSR, 2>(ramc, &sin);
    __tx_fence();
    cos_cache[i] = cos * mscale;
    sin_cache[i] = sin * mscale * sin_sign;

    theta *= theta_scale;
  }
}

__device__ long rope_f32_i32_f32(float* a, int* b, float* res,
                                 int bs, int h, int n, int m, int n_dims,
                                 uint32_t theta_scale_, uint32_t freq_scale_, int mdst) {
  float theta_scale = u32_as_float(theta_scale_);
  float freq_scale = u32_as_float(freq_scale_);
  const int vec_len = n_dims / 2;

  float pos_cos[m / 2];
  float pos_sin[m / 2];

  config_tcsr_vector<oneTCSR>(1);
  config_tcsr_vector<aTCSR>(m);
  config_tcsr_vector<bTCSR>(m);
  config_tcsr_vector<cTCSR>(m);
  config_tcsr_vector<dTCSR>(m);

  for (int i = 0; i < bs; ++i) { // batch
    for (int j = 0; j < h; ++j) { // seq-len
      int64_t p = b[j];
      rope_yarn(pos_cos, pos_sin, p, theta_scale, freq_scale, mdst);

      for (int k = 0; k < n; ++k) { // attn-heads
        const int64_t head_offset = (i * h * n + j * n + k) * m;
        float* x0_ptr = &a[head_offset];
        float* x1_ptr = &a[head_offset + vec_len];
        float* res0_ptr = &res[head_offset];
        float* res1_ptr = &res[head_offset + vec_len];

        // load stage
        __tx_async_uram_load<aTCSR, aTCSR, 2>(x0_ptr, (void*)uram0);
        __tx_async_uram_load<bTCSR, bTCSR, 2>(x1_ptr, (void*)uram1);
        __tx_async_uram_load<cTCSR, cTCSR, 2>(pos_cos, (void*)uram2);
        __tx_async_uram_load<dTCSR, dTCSR, 2>(pos_sin, (void*)uram3);
        __tx_fence();

        // compute stage
        __tx_fp_mul_fp32<aTCSR, cTCSR, aTCSR, 0b0000>((void*)uram0, (void*)uram2, (void*)uram4);  // buf1 = x0 * cos
        __tx_fp_mul_fp32<bTCSR, dTCSR, bTCSR, 0b0000>((void*)uram1, (void*)uram3, (void*)uram5);  // buf2 = x1 * sin
        __tx_fp_mul_fp32<aTCSR, dTCSR, aTCSR, 0b0000>((void*)uram0, (void*)uram3, (void*)uram6);  // buf3 = x0 * sin
        __tx_fp_mul_fp32<bTCSR, cTCSR, bTCSR, 0b0000>((void*)uram1, (void*)uram2, (void*)uram7);  // buf4 = x1 * cos
        __tx_fence();

        __tx_fp_sub_fp32<aTCSR, aTCSR, aTCSR, 0b0000>((void*)uram4, (void*)uram5, (void*)uram0);  // res0 = buf1 - buf2
        __tx_fp_add_fp32<aTCSR, aTCSR, aTCSR, 0b0000>((void*)uram6, (void*)uram7, (void*)uram1);  // res1 = buf3 + buf4
        __tx_fence();

        // store stage
        // fill res[i][j][k][:prope.n_dims]
        __tx_async_uram_store<aTCSR, aTCSR, 2>((void*)uram0, res0_ptr);
        __tx_async_uram_store<aTCSR, aTCSR, 2>((void*)uram1, res1_ptr);
        __tx_fence();

        // fill remain res[i][j][k][prope.n_dims:]
        const int64_t fill_res_idx = i * h * n * m + j * n * m + k * m;
        for (int p = n_dims; p < m; p += 2) {
          res[fill_res_idx + p] = a[fill_res_idx + p];
          res[fill_res_idx + p + 1] = a[fill_res_idx + p + 1];
        }
      }
    }
  }
  return 0;
}