#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const uint64_t nram0 = 0x1000000;
const uint64_t nram1 = 0x2000000;
const uint64_t nram2 = 0x3000000;

static inline void initTCSR(int m, int n, TCFG tcsr) {
  tcsr.dim0 = m;
  tcsr.stride0 = 1;
  tcsr.stride1 = m;
  tcsr.dim1 = n;
  tcsr.dim2 = 1;
  tcsr.stride2 = m * n;
  tcsr.dim3 = 1;
  tcsr.stride3 = m * n;
}

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

static inline void tx_sfu_exp_fp32_(void* input, void* output) {
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void*)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void*)nram1);
  __tx_fence();
  __tx_sfu_exp_fp32<0, 1>((void*)nram0, (void*)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void*)nram1, output);
  __tx_fence();
}


extern "C" void tx_fp_reduce_max_fp32(void* input0, void* dst, int m, int n, const int axis);
extern "C" void tx_fp_reduce_add_fp32(void* input0, void* dst, int m, int n, const int axis);

__device__ long softmax_f32_f32_f32(float* a, float* b, float* res, float scale,
              int bs, int h, int n, int m) {
  TCFG tcsr[2];

  for (int i = 0; i < bs; ++i) { // batch
    for (int j = 0; j < (h * n); ++j) { // attn-heads * seq_len
      float slope = 1.0f; // only support max_bias == 0.0f in qwen2

      configTCSRs(1, m);

      // scale
      float scale_vec[m];memset(scale_vec, scale, sizeof(scale_vec));
      float* a_ptr = &a[i * h * n * m + j * m];
      tx_fp_mul_fp32_(a_ptr, scale_vec, a_ptr); // a[i][j][k][:] *= scale


      // mask
      float buf1_vec[m];
      float slope_vec[m];memset(slope_vec, slope, sizeof(slope_vec));
      float* b_ptr = &b[i * h * n * m + (j % n) * m];
      tx_fp_mul_fp32_(b_ptr, slope_vec, buf1_vec); // a[i][j][k][:] += slope * b[i][0][k][:]
      tx_fp_mul_fp32_(a_ptr, buf1_vec, a_ptr); // a[i][j][k][:] += slope * b[i][0][k][:]

      // get max value for dim[-1]
      float maxval = -INFINITY;
      tx_fp_reduce_max_fp32(a_ptr, &maxval, m, 1, 1); // maxval = max(a[i][j][k][:])

      configTCSRs(1, m);
      float sum = 0.0f;
      float* res_ptr = &res[i * h * n * m + j * m];
      float maxval_vec[m]; memset(maxval_vec, maxval, sizeof(maxval_vec));
      tx_fp_sub_fp32_(a_ptr, maxval_vec, res_ptr);  // a[i][j][k][:] -= max
      tx_sfu_exp_fp32_(res_ptr, res_ptr);  // x(i) = e^(x(i) - max)
      tx_fp_reduce_add_fp32(res_ptr, &sum, m, 1, 1);  // sum = 1 / (sum of e^(x(i) - max))

      sum = 1.0 / sum;

      // x(i) = e^(x(i) - max) / (sum of e^(x(i) - max))
      float sum_vec[m];memset(sum_vec, sum, sizeof(sum_vec));
      tx_fp_mul_fp32_(res_ptr, sum_vec, res_ptr); // res[i][j][k][:] *= (float)sum
    }
  }
  return 0;
}