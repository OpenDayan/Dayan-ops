#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

const long convTCSR = 8;

const uint64_t uram0 = 0x1000000;
const uint64_t uram1 = 0x2000000;
const uint64_t uram2 = 0x3000000;

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

  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(tcsr[2].dim0);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(tcsr[2].stride0);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(tcsr[2].dim1);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(tcsr[2].stride1);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(tcsr[2].dim2);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(tcsr[2].stride2);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(tcsr[2].dim3);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(tcsr[2].stride3);

  __tx_tcsr_write<convTCSR * 8 + TCSR_DIM0>(tcsr[3].dim0);
  __tx_tcsr_write<convTCSR * 8 + TCSR_STRIDE0>(tcsr[3].stride0);
  __tx_tcsr_write<convTCSR * 8 + TCSR_DIM1>(tcsr[3].dim1);
  __tx_tcsr_write<convTCSR * 8 + TCSR_STRIDE1>(tcsr[3].stride1);
}

__device__ void tx_conv_float_fp32_fp32_fp32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {
  config_dims_and_strides(tcsr);
  __tx_async_uram_load<aTCSR, aTCSR, 2>(input0, (void *)uram0);
  __tx_async_uram_load<bTCSR, bTCSR, 2>(input1, (void *)uram1);
  __tx_async_uram_load<cTCSR, cTCSR, 2>(dst, (void *)uram2);
  __tx_fence();

  __tx_conv_float_f32_f32_f32<0, 1, 2>((void *)uram0, (void *)uram1, (void *)uram2);                                                        \
   
  __tx_fence();

  __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram2, dst);
  __tx_fence();
}

__device__ void tx_conv_float_fp16_fp16_fp32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {
  config_dims_and_strides(tcsr);
  __tx_async_uram_load<aTCSR, aTCSR, 1>(input0, (void *)uram0);
  __tx_async_uram_load<bTCSR, bTCSR, 1>(input1, (void *)uram1);
  __tx_async_uram_load<cTCSR, cTCSR, 2>(dst, (void *)uram2);
  __tx_fence();

  __tx_conv_float_fp16_fp16_f32<0, 1, 2>((void *)uram0, (void *)uram1, (void *)uram2);                                                        \
   
  __tx_fence();

  __tx_async_uram_store<cTCSR, cTCSR, 2>((void *)uram2, dst);
  __tx_fence();
}

__device__ void tx_conv_float_fp16_fp16_fp16(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {
  config_dims_and_strides(tcsr);
  __tx_async_uram_load<aTCSR, aTCSR, 1>(input0, (void *)uram0);
  __tx_async_uram_load<bTCSR, bTCSR, 1>(input1, (void *)uram1);
  __tx_async_uram_load<cTCSR, cTCSR, 1>(dst, (void *)uram2);
  __tx_fence();

  __tx_conv_float_fp16_fp16_fp16<0, 1, 2>((void *)uram0, (void *)uram1, (void *)uram2);                                                        \
   
  __tx_fence();

  __tx_async_uram_store<cTCSR, cTCSR, 1>((void *)uram2, dst);
  __tx_fence();
}