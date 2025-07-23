#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

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

__device__ long rope_f32_i32_f32(float* x, float* pos_sin, float* pos_cos, float* res,
                                 int bs, int h, int n, int m, int n_dims) {
  const int vec_len = n_dims / 2;
  configTCSRs(1, vec_len);

  float buf1_vec[vec_len];
  float buf2_vec[vec_len];
  float buf3_vec[vec_len];
  float buf4_vec[vec_len];

  for (int i = 0; i < bs; ++i) { // batch
    for (int j = 0; j < h; ++j) { // seq-len
      const int64_t pos_offset = (i * h + j) * vec_len;
      const float* cos_vec_ptr = &pos_cos[pos_offset];
      const float* sin_vec_ptr = &pos_sin[pos_offset];
      for (int k = 0; k < n; ++k) { // attn-heads
        // fill res[i][j][k][:prope.n_dims]
        const int64_t head_offset = (i * h * n + j * n + k) * m;

        const float* x0_ptr = &x[head_offset];
        const float* x1_ptr = &x[head_offset + vec_len];
        float* res0_ptr = &res[head_offset];
        float* res1_ptr = &res[head_offset + vec_len];
        // res0_ptr = x0_ptr * cos_vec_ptr - x1_ptr * sin_vec_ptr
        tx_fp_mul_fp32_((float*)x0_ptr, (float*)cos_vec_ptr, buf1_vec);      // buf1_vec = x0 * cos
        tx_fp_mul_fp32_((float*)x1_ptr, (float*)sin_vec_ptr, buf2_vec);      // buf2_vec = x1 * sin
        tx_fp_sub_fp32_(buf1_vec, buf2_vec, res0_ptr);       // res0 = buf1 - buf2

        // res1_ptr = x0_ptr * sin_vec_ptr + x1_ptr * cos_vec_ptr
        tx_fp_mul_fp32_((float*)x0_ptr, (float*)sin_vec_ptr, buf3_vec);      // buf3_vec = x0 * sin
        tx_fp_mul_fp32_((float*)x1_ptr, (float*)cos_vec_ptr, buf4_vec);      // buf4_vec = x1 * cos
        tx_fp_add_fp32_(buf3_vec, buf4_vec, res1_ptr);       // res1 = buf3 + buf4
      }
    }
  }
  return 0;
}