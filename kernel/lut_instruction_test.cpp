#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;

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

  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(tcsr[2].dim0);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(tcsr[2].stride0);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(tcsr[2].dim1);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(tcsr[2].stride1);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(tcsr[2].dim2);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(tcsr[2].stride2);
  __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(tcsr[2].dim3);
  __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(tcsr[2].stride3);
}

__device__ void tx_lut_int_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr, int width, int log2Length) {
  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();
  // width is 2, log2length is 3. lookuptable size is 8 x 2.
  __tx_lut_int_s32<aTCSR, 2, cTCSR, 3>((void *)nram0, (void *)nram1,  (void *)nram2);
  __tx_fence();
  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}