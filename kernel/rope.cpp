#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>

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


static inline void tx_fp_add_fp32_(void* input0, void* input1, void* dst) {
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void*)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void*)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void*)nram2);
  __tx_fence();

  __tx_fp_add_fp32<0, 1, 2, 0b0000>((void*)nram0, (void*)nram1, (void*)nram2);

  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram2, dst);
  __tx_fence();
}


static inline void tx_fp_sub_fp32_(void* input0, void* input1, void* dst) {
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void*)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void*)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void*)nram2);
  __tx_fence();

  __tx_fp_sub_fp32<0, 1, 2, 0b0000>((void*)nram0, (void*)nram1, (void*)nram2);

  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram2, dst);
  __tx_fence();
}

static inline void tx_fp_mul_fp32_(void* input0, void* input1, void* dst) {
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void*)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void*)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void*)nram2);
  __tx_fence();

  __tx_fp_mul_fp32<0, 1, 2, 0b0000>((void*)nram0, (void*)nram1, (void*)nram2);

  __tx_fence();
  __tx_async_nram_store<cTCSR, cTCSR, 2>((void*)nram2, dst);
  __tx_fence();
}

static inline void tx_sfu_sin_fp32_(void* input, void* output) {
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void*)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void*)nram1);
  __tx_fence();
  __tx_sfu_sin_fp32<0, 1>((void*)nram0, (void*)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void*)nram1, output);
  __tx_fence();
}


static inline void tx_sfu_cos_fp32_(void* input, void* output) {
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void*)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void*)nram1);
  __tx_fence();
  __tx_sfu_cos_fp32<0, 1>((void*)nram0, (void*)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void*)nram1, output);
  __tx_fence();
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
  configTCSRs(1, 1);
  float theta = theta_base;
  for (int i = 0; i < m / 2; i++) {
    // const float ff = freq_factors ? freq_factors[i/2] : 1.0f;
    const float ff = 1.0f;

    // YaRN
    float theta_extrap = theta / ff;
    float theta_interp = freq_scale * (theta / ff);
    // if (ext_factor != 0.0f) ...
    float cos;
    float sin;
    tx_sfu_cos_fp32_(&theta_interp, &cos);
    tx_sfu_sin_fp32_(&theta_interp, &sin);
    cos_cache[i] = cos * mscale;
    sin_cache[i] = sin * mscale * sin_sign;

    theta *= theta_scale;
  }
}

__device__ long rope_f32_i32_f32(float* a, int* b, float* res,
                                 int bs, int h, int n, int m, int n_dims,
                                 float theta_scale, float freq_scale, int mdst) {
  const int vec_len = n_dims / 2;
  float buf1_vec[vec_len];
  float buf2_vec[vec_len];
  float buf3_vec[vec_len];
  float buf4_vec[vec_len];

  float pos_cos[m / 2];
  float pos_sin[m / 2];

  for (int i = 0; i < bs; ++i) { // batch
    for (int j = 0; j < h; ++j) { // seq-len
      int64_t p = b[j];
      rope_yarn(pos_cos, pos_sin, p, theta_scale, freq_scale, mdst);

      configTCSRs(1, vec_len);
      for (int k = 0; k < n; ++k) { // attn-heads
        // fill res[i][j][k][:prope.n_dims]
        const int64_t head_offset = (i * h * n + j * n + k) * m;
        const float* x0_ptr = &a[head_offset];
        const float* x1_ptr = &a[head_offset + vec_len];
        float* res0_ptr = &res[head_offset];
        float* res1_ptr = &res[head_offset + vec_len];
        // res0_ptr = x0_ptr * pos_cos - x1_ptr * pos_sin
        tx_fp_mul_fp32_((float*)x0_ptr, (float*)pos_cos, buf1_vec);      // buf1_vec = x0 * cos
        tx_fp_mul_fp32_((float*)x1_ptr, (float*)pos_sin, buf2_vec);      // buf2_vec = x1 * sin
        tx_fp_sub_fp32_(buf1_vec, buf2_vec, res0_ptr);       // res0 = buf1 - buf2

        // res1_ptr = x0_ptr * pos_sin + x1_ptr * pos_cos
        tx_fp_mul_fp32_((float*)x0_ptr, (float*)pos_sin, buf3_vec);      // buf3_vec = x0 * sin
        tx_fp_mul_fp32_((float*)x1_ptr, (float*)pos_cos, buf4_vec);      // buf4_vec = x1 * cos
        tx_fp_add_fp32_(buf3_vec, buf4_vec, res1_ptr);       // res1 = buf3 + buf4

        // fill remain res[i][j][k][prope.n_dims:]
        for (int p = n_dims; p < m; p += 2) {
          res[i * h * n * m + j * n * m + k * m + p] = a[i * h * n * m + j * n * m + k * m + p];
          res[i * h * n * m + j * n * m + k * m + p + 1] = a[i * h * n * m + j * n * m + k * m + p + 1];
        }
      }
    }
  }
  return 0;
}