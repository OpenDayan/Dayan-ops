#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCSR = 0;
const long bTCSR = 1;

const uint64_t nram0 = 0x1000000;
const uint64_t nram1 = 0x2000000;
const uint64_t nram2 = 0x3000000;

static inline void config_dims_and_strides(TCFG *tcsr) {
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(tcsr[0].dim0);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(tcsr[0].stride0);
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(tcsr[0].dim1);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(tcsr[0].stride1);
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(tcsr[0].dim2);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(tcsr[0].stride2);
  __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(tcsr[0].dim3);
  __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(tcsr[0].stride3);

  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(tcsr[1].dim0);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(tcsr[1].stride0);
  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(tcsr[1].dim1);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(tcsr[1].stride1);
  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(tcsr[1].dim2);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(tcsr[1].stride2);
  __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(tcsr[1].dim3);
  __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(tcsr[1].stride3);
}

__device__ void tx_sfu_recip_fp32(void *input, void *output,
                                      TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void *)nram1);
  __tx_fence();
  __tx_sfu_recip_fp32<0, 1>((void *)nram0, (void *)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram1, output);
  __tx_fence();
}

__device__ void tx_sfu_rsqrt_fp32(void *input, void *output,
                                      TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void *)nram1);
  __tx_fence();
  __tx_sfu_rsqrt_fp32<0, 1>((void *)nram0, (void *)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram1, output);
  __tx_fence();
}

__device__ void tx_sfu_sin_fp32(void *input, void *output,
                                      TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void *)nram1);
  __tx_fence();
  __tx_sfu_sin_fp32<0, 1>((void *)nram0, (void *)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram1, output);
  __tx_fence();
}

__device__ void tx_sfu_cos_fp32(void *input, void *output,
                                      TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void *)nram1);
  __tx_fence();
  __tx_sfu_cos_fp32<0, 1>((void *)nram0, (void *)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram1, output);
  __tx_fence();
}

__device__ void tx_sfu_exp_fp32(void *input, void *output,
                                      TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(output, (void *)nram1);
  __tx_fence();
  __tx_sfu_exp_fp32<0, 1>((void *)nram0, (void *)nram1);
  __tx_fence();

  __tx_async_nram_store<bTCSR, bTCSR, 2>((void *)nram1, output);
  __tx_fence();
}
