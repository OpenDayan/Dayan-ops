#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

__device__ void matmul_fp32_fp32_fp32(void *input0, void *input1, void *dst, int m, int n,
                                        int k) {
  
  // using TCSR0, TCSR1, TCSR2 to configure the dims.
  __tx_tcsr_write<0>(m);
  __tx_tcsr_write<16>(m);
  __tx_tcsr_write<2>(k);
  __tx_tcsr_write<8>(k);
  __tx_tcsr_write<10>(n);
  __tx_tcsr_write<18>(n);

  __tx_fence();
  // TODO: copy inputs to nram
 
  __tx_matmul_float_f32_f32_f32<0, 1, 2>(
              (void *)input0, (void *)input1, (void *)dst);
  // TODO: copy output from nram
  __tx_fence();
}
