#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <cstddef>
#include <stddef.h>
#include <stdint.h>

__device__ void tx_int_add_s32(void *input0, void *input1, void *dst, int m,
                               int n) {
  // using TCSR0, TCSR1, TCSR2 to configure the dims and strides.
  __tx_tcsr_write<0>(m); //dim0
  __tx_tcsr_write<1>(n); //stride0
  __tx_tcsr_write<2>(n); //dim1
  __tx_tcsr_write<3>(1); //stride1
  __tx_tcsr_write<4>(1); //dim2
  __tx_tcsr_write<5>(1); //stride2
  __tx_tcsr_write<6>(1); //dim3
  __tx_tcsr_write<7>(1); //stride3

  __tx_tcsr_write<8>(m); //dim0
  __tx_tcsr_write<9>(n); //stride0
  __tx_tcsr_write<10>(n); //dim1
  __tx_tcsr_write<11>(1); //stride1
  __tx_tcsr_write<12>(1); //dim2
  __tx_tcsr_write<13>(1); //stride2
  __tx_tcsr_write<14>(1); //dim3
  __tx_tcsr_write<15>(1); //stride3

  __tx_tcsr_write<16>(m); //dim0
  __tx_tcsr_write<17>(n); //stride0
  __tx_tcsr_write<18>(n); //dim1
  __tx_tcsr_write<19>(1); //stride1
  __tx_tcsr_write<20>(1); //dim2
  __tx_tcsr_write<21>(1); //stride2
  __tx_tcsr_write<22>(1); //dim3
  __tx_tcsr_write<23>(1); //stride3

  __tx_fence();
  // TODO: copy inputs to nram
  // tx.int.add.s32
  __tx_int_add_s32<0, 1, 2, 0>((void *)input0, (void *)input1, (void *)dst);
  __tx_fence();
  // TODO: copy output from nram
}