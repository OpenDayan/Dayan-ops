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

static inline uint8_t get_broadcast(TCFG *tcsr) {
  uint8_t broadcast = 0;

  if (tcsr[0].dim0 != tcsr[1].dim0 &&
      (tcsr[0].dim0 == 1 || tcsr[1].dim0 == 1)) {
    broadcast |= 0b0001;
  }

  if (tcsr[0].dim1 != tcsr[1].dim1 &&
      (tcsr[0].dim1 == 1 || tcsr[1].dim1 == 1)) {
    broadcast |= 0b0010;
  }

  if (tcsr[0].dim2 != tcsr[1].dim2 &&
      (tcsr[0].dim2 == 1 || tcsr[1].dim2 == 1)) {
    broadcast |= 0b0100;
  }

  if (tcsr[0].dim3 != tcsr[1].dim3 &&
      (tcsr[0].dim3 == 1 || tcsr[1].dim3 == 1)) {
    broadcast |= 0b1000;
  }

  return broadcast;
}

__device__ void tx_int_add_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {
  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);

#define CALL_TX_INT_ADD_S32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_int_add_s32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;

  switch (broadcast) {
    CALL_TX_INT_ADD_S32(0b0000) // 0
    CALL_TX_INT_ADD_S32(0b0001) // 1
    CALL_TX_INT_ADD_S32(0b0010) // 2
    CALL_TX_INT_ADD_S32(0b0011) // 3
    CALL_TX_INT_ADD_S32(0b0100) // 4
    CALL_TX_INT_ADD_S32(0b0101) // 5
    CALL_TX_INT_ADD_S32(0b0110) // 6
    CALL_TX_INT_ADD_S32(0b0111) // 7
    CALL_TX_INT_ADD_S32(0b1000) // 8
    CALL_TX_INT_ADD_S32(0b1001) // 9
    CALL_TX_INT_ADD_S32(0b1010) // 10
    CALL_TX_INT_ADD_S32(0b1011) // 11
    CALL_TX_INT_ADD_S32(0b1100) // 12
    CALL_TX_INT_ADD_S32(0b1101) // 13
    CALL_TX_INT_ADD_S32(0b1110) // 14
    CALL_TX_INT_ADD_S32(0b1111) // 15
  default:
    break;
  }
#undef CALL_TX_INT_ADD_S32

  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sub_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {
  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_INT_SUB_S32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_int_sub_s32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_INT_SUB_S32(0b0000) // 0
    CALL_TX_INT_SUB_S32(0b0001) // 1
    CALL_TX_INT_SUB_S32(0b0010) // 2
    CALL_TX_INT_SUB_S32(0b0011) // 3
    CALL_TX_INT_SUB_S32(0b0100) // 4
    CALL_TX_INT_SUB_S32(0b0101) // 5
    CALL_TX_INT_SUB_S32(0b0110) // 6
    CALL_TX_INT_SUB_S32(0b0111) // 7
    CALL_TX_INT_SUB_S32(0b1000) // 8
    CALL_TX_INT_SUB_S32(0b1001) // 9
    CALL_TX_INT_SUB_S32(0b1010) // 10
    CALL_TX_INT_SUB_S32(0b1011) // 11
    CALL_TX_INT_SUB_S32(0b1100) // 12
    CALL_TX_INT_SUB_S32(0b1101) // 13
    CALL_TX_INT_SUB_S32(0b1110) // 14
    CALL_TX_INT_SUB_S32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_INT_SUB_S32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_mul_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_INT_MUL_S32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_int_mul_s32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_INT_MUL_S32(0b0000) // 0
    CALL_TX_INT_MUL_S32(0b0001) // 1
    CALL_TX_INT_MUL_S32(0b0010) // 2
    CALL_TX_INT_MUL_S32(0b0011) // 3
    CALL_TX_INT_MUL_S32(0b0100) // 4
    CALL_TX_INT_MUL_S32(0b0101) // 5
    CALL_TX_INT_MUL_S32(0b0110) // 6
    CALL_TX_INT_MUL_S32(0b0111) // 7
    CALL_TX_INT_MUL_S32(0b1000) // 8
    CALL_TX_INT_MUL_S32(0b1001) // 9
    CALL_TX_INT_MUL_S32(0b1010) // 10
    CALL_TX_INT_MUL_S32(0b1011) // 11
    CALL_TX_INT_MUL_S32(0b1100) // 12
    CALL_TX_INT_MUL_S32(0b1101) // 13
    CALL_TX_INT_MUL_S32(0b1110) // 14
    CALL_TX_INT_MUL_S32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_INT_MUL_S32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_srl_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_INT_SRL_S32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_int_srl_s32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_INT_SRL_S32(0b0000) // 0
    CALL_TX_INT_SRL_S32(0b0001) // 1
    CALL_TX_INT_SRL_S32(0b0010) // 2
    CALL_TX_INT_SRL_S32(0b0011) // 3
    CALL_TX_INT_SRL_S32(0b0100) // 4
    CALL_TX_INT_SRL_S32(0b0101) // 5
    CALL_TX_INT_SRL_S32(0b0110) // 6
    CALL_TX_INT_SRL_S32(0b0111) // 7
    CALL_TX_INT_SRL_S32(0b1000) // 8
    CALL_TX_INT_SRL_S32(0b1001) // 9
    CALL_TX_INT_SRL_S32(0b1010) // 10
    CALL_TX_INT_SRL_S32(0b1011) // 11
    CALL_TX_INT_SRL_S32(0b1100) // 12
    CALL_TX_INT_SRL_S32(0b1101) // 13
    CALL_TX_INT_SRL_S32(0b1110) // 14
    CALL_TX_INT_SRL_S32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_INT_SRL_S32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sra_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_INT_SRA_S32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_int_sra_s32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_INT_SRA_S32(0b0000) // 0
    CALL_TX_INT_SRA_S32(0b0001) // 1
    CALL_TX_INT_SRA_S32(0b0010) // 2
    CALL_TX_INT_SRA_S32(0b0011) // 3
    CALL_TX_INT_SRA_S32(0b0100) // 4
    CALL_TX_INT_SRA_S32(0b0101) // 5
    CALL_TX_INT_SRA_S32(0b0110) // 6
    CALL_TX_INT_SRA_S32(0b0111) // 7
    CALL_TX_INT_SRA_S32(0b1000) // 8
    CALL_TX_INT_SRA_S32(0b1001) // 9
    CALL_TX_INT_SRA_S32(0b1010) // 10
    CALL_TX_INT_SRA_S32(0b1011) // 11
    CALL_TX_INT_SRA_S32(0b1100) // 12
    CALL_TX_INT_SRA_S32(0b1101) // 13
    CALL_TX_INT_SRA_S32(0b1110) // 14
    CALL_TX_INT_SRA_S32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_INT_SRA_S32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_sll_s32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_INT_SLL_S32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_int_sll_s32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_INT_SLL_S32(0b0000) // 0
    CALL_TX_INT_SLL_S32(0b0001) // 1
    CALL_TX_INT_SLL_S32(0b0010) // 2
    CALL_TX_INT_SLL_S32(0b0011) // 3
    CALL_TX_INT_SLL_S32(0b0100) // 4
    CALL_TX_INT_SLL_S32(0b0101) // 5
    CALL_TX_INT_SLL_S32(0b0110) // 6
    CALL_TX_INT_SLL_S32(0b0111) // 7
    CALL_TX_INT_SLL_S32(0b1000) // 8
    CALL_TX_INT_SLL_S32(0b1001) // 9
    CALL_TX_INT_SLL_S32(0b1010) // 10
    CALL_TX_INT_SLL_S32(0b1011) // 11
    CALL_TX_INT_SLL_S32(0b1100) // 12
    CALL_TX_INT_SLL_S32(0b1101) // 13
    CALL_TX_INT_SLL_S32(0b1110) // 14
    CALL_TX_INT_SLL_S32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_INT_SLL_S32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_int_lutop_s32(void *input0, void *input1, void *dst,
                                 uint8_t lutop, TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();
  lutop = lutop & 0b1111;

  uint8_t broadcast = get_broadcast(tcsr);
  broadcast &= 0b1111;

#define CASE_LUTOP_BROADCAST(l, b)                                             \
  case ((l << 4) | b):                                                         \
    __tx_int_lutop_s32<l, 0, 1, 2, b>((void *)nram0, (void *)nram1,            \
                                      (void *)nram2);                          \
    break;

  switch ((lutop << 4) | broadcast) {
    CASE_LUTOP_BROADCAST(0, 0)
    CASE_LUTOP_BROADCAST(0, 1)
    CASE_LUTOP_BROADCAST(0, 2)
    CASE_LUTOP_BROADCAST(0, 3)
    CASE_LUTOP_BROADCAST(0, 4)
    CASE_LUTOP_BROADCAST(0, 5)
    CASE_LUTOP_BROADCAST(0, 6)
    CASE_LUTOP_BROADCAST(0, 7)
    CASE_LUTOP_BROADCAST(0, 8)
    CASE_LUTOP_BROADCAST(0, 9)
    CASE_LUTOP_BROADCAST(0, 10)
    CASE_LUTOP_BROADCAST(0, 11)
    CASE_LUTOP_BROADCAST(0, 12)
    CASE_LUTOP_BROADCAST(0, 13)
    CASE_LUTOP_BROADCAST(0, 14)
    CASE_LUTOP_BROADCAST(0, 15)
    CASE_LUTOP_BROADCAST(1, 0)
    CASE_LUTOP_BROADCAST(1, 1)
    CASE_LUTOP_BROADCAST(1, 2)
    CASE_LUTOP_BROADCAST(1, 3)
    CASE_LUTOP_BROADCAST(1, 4)
    CASE_LUTOP_BROADCAST(1, 5)
    CASE_LUTOP_BROADCAST(1, 6)
    CASE_LUTOP_BROADCAST(1, 7)
    CASE_LUTOP_BROADCAST(1, 8)
    CASE_LUTOP_BROADCAST(1, 9)
    CASE_LUTOP_BROADCAST(1, 10)
    CASE_LUTOP_BROADCAST(1, 11)
    CASE_LUTOP_BROADCAST(1, 12)
    CASE_LUTOP_BROADCAST(1, 13)
    CASE_LUTOP_BROADCAST(1, 14)
    CASE_LUTOP_BROADCAST(1, 15)
    CASE_LUTOP_BROADCAST(2, 0)
    CASE_LUTOP_BROADCAST(2, 1)
    CASE_LUTOP_BROADCAST(2, 2)
    CASE_LUTOP_BROADCAST(2, 3)
    CASE_LUTOP_BROADCAST(2, 4)
    CASE_LUTOP_BROADCAST(2, 5)
    CASE_LUTOP_BROADCAST(2, 6)
    CASE_LUTOP_BROADCAST(2, 7)
    CASE_LUTOP_BROADCAST(2, 8)
    CASE_LUTOP_BROADCAST(2, 9)
    CASE_LUTOP_BROADCAST(2, 10)
    CASE_LUTOP_BROADCAST(2, 11)
    CASE_LUTOP_BROADCAST(2, 12)
    CASE_LUTOP_BROADCAST(2, 13)
    CASE_LUTOP_BROADCAST(2, 14)
    CASE_LUTOP_BROADCAST(2, 15)
    CASE_LUTOP_BROADCAST(3, 0)
    CASE_LUTOP_BROADCAST(3, 1)
    CASE_LUTOP_BROADCAST(3, 2)
    CASE_LUTOP_BROADCAST(3, 3)
    CASE_LUTOP_BROADCAST(3, 4)
    CASE_LUTOP_BROADCAST(3, 5)
    CASE_LUTOP_BROADCAST(3, 6)
    CASE_LUTOP_BROADCAST(3, 7)
    CASE_LUTOP_BROADCAST(3, 8)
    CASE_LUTOP_BROADCAST(3, 9)
    CASE_LUTOP_BROADCAST(3, 10)
    CASE_LUTOP_BROADCAST(3, 11)
    CASE_LUTOP_BROADCAST(3, 12)
    CASE_LUTOP_BROADCAST(3, 13)
    CASE_LUTOP_BROADCAST(3, 14)
    CASE_LUTOP_BROADCAST(3, 15)
    CASE_LUTOP_BROADCAST(4, 0)
    CASE_LUTOP_BROADCAST(4, 1)
    CASE_LUTOP_BROADCAST(4, 2)
    CASE_LUTOP_BROADCAST(4, 3)
    CASE_LUTOP_BROADCAST(4, 4)
    CASE_LUTOP_BROADCAST(4, 5)
    CASE_LUTOP_BROADCAST(4, 6)
    CASE_LUTOP_BROADCAST(4, 7)
    CASE_LUTOP_BROADCAST(4, 8)
    CASE_LUTOP_BROADCAST(4, 9)
    CASE_LUTOP_BROADCAST(4, 10)
    CASE_LUTOP_BROADCAST(4, 11)
    CASE_LUTOP_BROADCAST(4, 12)
    CASE_LUTOP_BROADCAST(4, 13)
    CASE_LUTOP_BROADCAST(4, 14)
    CASE_LUTOP_BROADCAST(4, 15)
    CASE_LUTOP_BROADCAST(5, 0)
    CASE_LUTOP_BROADCAST(5, 1)
    CASE_LUTOP_BROADCAST(5, 2)
    CASE_LUTOP_BROADCAST(5, 3)
    CASE_LUTOP_BROADCAST(5, 4)
    CASE_LUTOP_BROADCAST(5, 5)
    CASE_LUTOP_BROADCAST(5, 6)
    CASE_LUTOP_BROADCAST(5, 7)
    CASE_LUTOP_BROADCAST(5, 8)
    CASE_LUTOP_BROADCAST(5, 9)
    CASE_LUTOP_BROADCAST(5, 10)
    CASE_LUTOP_BROADCAST(5, 11)
    CASE_LUTOP_BROADCAST(5, 12)
    CASE_LUTOP_BROADCAST(5, 13)
    CASE_LUTOP_BROADCAST(5, 14)
    CASE_LUTOP_BROADCAST(5, 15)
    CASE_LUTOP_BROADCAST(6, 0)
    CASE_LUTOP_BROADCAST(6, 1)
    CASE_LUTOP_BROADCAST(6, 2)
    CASE_LUTOP_BROADCAST(6, 3)
    CASE_LUTOP_BROADCAST(6, 4)
    CASE_LUTOP_BROADCAST(6, 5)
    CASE_LUTOP_BROADCAST(6, 6)
    CASE_LUTOP_BROADCAST(6, 7)
    CASE_LUTOP_BROADCAST(6, 8)
    CASE_LUTOP_BROADCAST(6, 9)
    CASE_LUTOP_BROADCAST(6, 10)
    CASE_LUTOP_BROADCAST(6, 11)
    CASE_LUTOP_BROADCAST(6, 12)
    CASE_LUTOP_BROADCAST(6, 13)
    CASE_LUTOP_BROADCAST(6, 14)
    CASE_LUTOP_BROADCAST(6, 15)
    CASE_LUTOP_BROADCAST(7, 0)
    CASE_LUTOP_BROADCAST(7, 1)
    CASE_LUTOP_BROADCAST(7, 2)
    CASE_LUTOP_BROADCAST(7, 3)
    CASE_LUTOP_BROADCAST(7, 4)
    CASE_LUTOP_BROADCAST(7, 5)
    CASE_LUTOP_BROADCAST(7, 6)
    CASE_LUTOP_BROADCAST(7, 7)
    CASE_LUTOP_BROADCAST(7, 8)
    CASE_LUTOP_BROADCAST(7, 9)
    CASE_LUTOP_BROADCAST(7, 10)
    CASE_LUTOP_BROADCAST(7, 11)
    CASE_LUTOP_BROADCAST(7, 12)
    CASE_LUTOP_BROADCAST(7, 13)
    CASE_LUTOP_BROADCAST(7, 14)
    CASE_LUTOP_BROADCAST(7, 15)
    CASE_LUTOP_BROADCAST(8, 0)
    CASE_LUTOP_BROADCAST(8, 1)
    CASE_LUTOP_BROADCAST(8, 2)
    CASE_LUTOP_BROADCAST(8, 3)
    CASE_LUTOP_BROADCAST(8, 4)
    CASE_LUTOP_BROADCAST(8, 5)
    CASE_LUTOP_BROADCAST(8, 6)
    CASE_LUTOP_BROADCAST(8, 7)
    CASE_LUTOP_BROADCAST(8, 8)
    CASE_LUTOP_BROADCAST(8, 9)
    CASE_LUTOP_BROADCAST(8, 10)
    CASE_LUTOP_BROADCAST(8, 11)
    CASE_LUTOP_BROADCAST(8, 12)
    CASE_LUTOP_BROADCAST(8, 13)
    CASE_LUTOP_BROADCAST(8, 14)
    CASE_LUTOP_BROADCAST(8, 15)
    CASE_LUTOP_BROADCAST(9, 0)
    CASE_LUTOP_BROADCAST(9, 1)
    CASE_LUTOP_BROADCAST(9, 2)
    CASE_LUTOP_BROADCAST(9, 3)
    CASE_LUTOP_BROADCAST(9, 4)
    CASE_LUTOP_BROADCAST(9, 5)
    CASE_LUTOP_BROADCAST(9, 6)
    CASE_LUTOP_BROADCAST(9, 7)
    CASE_LUTOP_BROADCAST(9, 8)
    CASE_LUTOP_BROADCAST(9, 9)
    CASE_LUTOP_BROADCAST(9, 10)
    CASE_LUTOP_BROADCAST(9, 11)
    CASE_LUTOP_BROADCAST(9, 12)
    CASE_LUTOP_BROADCAST(9, 13)
    CASE_LUTOP_BROADCAST(9, 14)
    CASE_LUTOP_BROADCAST(9, 15)
    CASE_LUTOP_BROADCAST(10, 0)
    CASE_LUTOP_BROADCAST(10, 1)
    CASE_LUTOP_BROADCAST(10, 2)
    CASE_LUTOP_BROADCAST(10, 3)
    CASE_LUTOP_BROADCAST(10, 4)
    CASE_LUTOP_BROADCAST(10, 5)
    CASE_LUTOP_BROADCAST(10, 6)
    CASE_LUTOP_BROADCAST(10, 7)
    CASE_LUTOP_BROADCAST(10, 8)
    CASE_LUTOP_BROADCAST(10, 9)
    CASE_LUTOP_BROADCAST(10, 10)
    CASE_LUTOP_BROADCAST(10, 11)
    CASE_LUTOP_BROADCAST(10, 12)
    CASE_LUTOP_BROADCAST(10, 13)
    CASE_LUTOP_BROADCAST(10, 14)
    CASE_LUTOP_BROADCAST(10, 15)
    CASE_LUTOP_BROADCAST(11, 0)
    CASE_LUTOP_BROADCAST(11, 1)
    CASE_LUTOP_BROADCAST(11, 2)
    CASE_LUTOP_BROADCAST(11, 3)
    CASE_LUTOP_BROADCAST(11, 4)
    CASE_LUTOP_BROADCAST(11, 5)
    CASE_LUTOP_BROADCAST(11, 6)
    CASE_LUTOP_BROADCAST(11, 7)
    CASE_LUTOP_BROADCAST(11, 8)
    CASE_LUTOP_BROADCAST(11, 9)
    CASE_LUTOP_BROADCAST(11, 10)
    CASE_LUTOP_BROADCAST(11, 11)
    CASE_LUTOP_BROADCAST(11, 12)
    CASE_LUTOP_BROADCAST(11, 13)
    CASE_LUTOP_BROADCAST(11, 14)
    CASE_LUTOP_BROADCAST(11, 15)
    CASE_LUTOP_BROADCAST(12, 0)
    CASE_LUTOP_BROADCAST(12, 1)
    CASE_LUTOP_BROADCAST(12, 2)
    CASE_LUTOP_BROADCAST(12, 3)
    CASE_LUTOP_BROADCAST(12, 4)
    CASE_LUTOP_BROADCAST(12, 5)
    CASE_LUTOP_BROADCAST(12, 6)
    CASE_LUTOP_BROADCAST(12, 7)
    CASE_LUTOP_BROADCAST(12, 8)
    CASE_LUTOP_BROADCAST(12, 9)
    CASE_LUTOP_BROADCAST(12, 10)
    CASE_LUTOP_BROADCAST(12, 11)
    CASE_LUTOP_BROADCAST(12, 12)
    CASE_LUTOP_BROADCAST(12, 13)
    CASE_LUTOP_BROADCAST(12, 14)
    CASE_LUTOP_BROADCAST(12, 15)
    CASE_LUTOP_BROADCAST(13, 0)
    CASE_LUTOP_BROADCAST(13, 1)
    CASE_LUTOP_BROADCAST(13, 2)
    CASE_LUTOP_BROADCAST(13, 3)
    CASE_LUTOP_BROADCAST(13, 4)
    CASE_LUTOP_BROADCAST(13, 5)
    CASE_LUTOP_BROADCAST(13, 6)
    CASE_LUTOP_BROADCAST(13, 7)
    CASE_LUTOP_BROADCAST(13, 8)
    CASE_LUTOP_BROADCAST(13, 9)
    CASE_LUTOP_BROADCAST(13, 10)
    CASE_LUTOP_BROADCAST(13, 11)
    CASE_LUTOP_BROADCAST(13, 12)
    CASE_LUTOP_BROADCAST(13, 13)
    CASE_LUTOP_BROADCAST(13, 14)
    CASE_LUTOP_BROADCAST(13, 15)
    CASE_LUTOP_BROADCAST(14, 0)
    CASE_LUTOP_BROADCAST(14, 1)
    CASE_LUTOP_BROADCAST(14, 2)
    CASE_LUTOP_BROADCAST(14, 3)
    CASE_LUTOP_BROADCAST(14, 4)
    CASE_LUTOP_BROADCAST(14, 5)
    CASE_LUTOP_BROADCAST(14, 6)
    CASE_LUTOP_BROADCAST(14, 7)
    CASE_LUTOP_BROADCAST(14, 8)
    CASE_LUTOP_BROADCAST(14, 9)
    CASE_LUTOP_BROADCAST(14, 10)
    CASE_LUTOP_BROADCAST(14, 11)
    CASE_LUTOP_BROADCAST(14, 12)
    CASE_LUTOP_BROADCAST(14, 13)
    CASE_LUTOP_BROADCAST(14, 14)
    CASE_LUTOP_BROADCAST(14, 15)
    CASE_LUTOP_BROADCAST(15, 0)
    CASE_LUTOP_BROADCAST(15, 1)
    CASE_LUTOP_BROADCAST(15, 2)
    CASE_LUTOP_BROADCAST(15, 3)
    CASE_LUTOP_BROADCAST(15, 4)
    CASE_LUTOP_BROADCAST(15, 5)
    CASE_LUTOP_BROADCAST(15, 6)
    CASE_LUTOP_BROADCAST(15, 7)
    CASE_LUTOP_BROADCAST(15, 8)
    CASE_LUTOP_BROADCAST(15, 9)
    CASE_LUTOP_BROADCAST(15, 10)
    CASE_LUTOP_BROADCAST(15, 11)
    CASE_LUTOP_BROADCAST(15, 12)
    CASE_LUTOP_BROADCAST(15, 13)
    CASE_LUTOP_BROADCAST(15, 14)
    CASE_LUTOP_BROADCAST(15, 15)
  default:
    break;
  }
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_add_fp32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_FP_ADD_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_add_fp32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_FP_ADD_FP32(0b0000) // 0
    CALL_TX_FP_ADD_FP32(0b0001) // 1
    CALL_TX_FP_ADD_FP32(0b0010) // 2
    CALL_TX_FP_ADD_FP32(0b0011) // 3
    CALL_TX_FP_ADD_FP32(0b0100) // 4
    CALL_TX_FP_ADD_FP32(0b0101) // 5
    CALL_TX_FP_ADD_FP32(0b0110) // 6
    CALL_TX_FP_ADD_FP32(0b0111) // 7
    CALL_TX_FP_ADD_FP32(0b1000) // 8
    CALL_TX_FP_ADD_FP32(0b1001) // 9
    CALL_TX_FP_ADD_FP32(0b1010) // 10
    CALL_TX_FP_ADD_FP32(0b1011) // 11
    CALL_TX_FP_ADD_FP32(0b1100) // 12
    CALL_TX_FP_ADD_FP32(0b1101) // 13
    CALL_TX_FP_ADD_FP32(0b1110) // 14
    CALL_TX_FP_ADD_FP32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_FP_ADD_FP32

  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_sub_fp32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_FP_SUB_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_sub_fp32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_FP_SUB_FP32(0b0000) // 0
    CALL_TX_FP_SUB_FP32(0b0001) // 1
    CALL_TX_FP_SUB_FP32(0b0010) // 2
    CALL_TX_FP_SUB_FP32(0b0011) // 3
    CALL_TX_FP_SUB_FP32(0b0100) // 4
    CALL_TX_FP_SUB_FP32(0b0101) // 5
    CALL_TX_FP_SUB_FP32(0b0110) // 6
    CALL_TX_FP_SUB_FP32(0b0111) // 7
    CALL_TX_FP_SUB_FP32(0b1000) // 8
    CALL_TX_FP_SUB_FP32(0b1001) // 9
    CALL_TX_FP_SUB_FP32(0b1010) // 10
    CALL_TX_FP_SUB_FP32(0b1011) // 11
    CALL_TX_FP_SUB_FP32(0b1100) // 12
    CALL_TX_FP_SUB_FP32(0b1101) // 13
    CALL_TX_FP_SUB_FP32(0b1110) // 14
    CALL_TX_FP_SUB_FP32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_FP_ADD_FP32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_mul_fp32(void *input0, void *input1, void *dst,
                               TCFG *tcsr) {

  config_dims_and_strides(tcsr);
  __tx_async_nram_load<aTCSR, aTCSR, 2>(input0, (void *)nram0);
  __tx_async_nram_load<bTCSR, bTCSR, 2>(input1, (void *)nram1);
  __tx_async_nram_load<cTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  uint8_t broadcast = get_broadcast(tcsr);
#define CALL_TX_FP_MUL_FP32(broadcast_val)                                     \
  case broadcast_val:                                                          \
    __tx_fp_mul_fp32<0, 1, 2, broadcast_val>((void *)nram0, (void *)nram1,     \
                                             (void *)nram2);                   \
    break;
  switch (broadcast) {
    CALL_TX_FP_MUL_FP32(0b0000) // 0
    CALL_TX_FP_MUL_FP32(0b0001) // 1
    CALL_TX_FP_MUL_FP32(0b0010) // 2
    CALL_TX_FP_MUL_FP32(0b0011) // 3
    CALL_TX_FP_MUL_FP32(0b0100) // 4
    CALL_TX_FP_MUL_FP32(0b0101) // 5
    CALL_TX_FP_MUL_FP32(0b0110) // 6
    CALL_TX_FP_MUL_FP32(0b0111) // 7
    CALL_TX_FP_MUL_FP32(0b1000) // 8
    CALL_TX_FP_MUL_FP32(0b1001) // 9
    CALL_TX_FP_MUL_FP32(0b1010) // 10
    CALL_TX_FP_MUL_FP32(0b1011) // 11
    CALL_TX_FP_MUL_FP32(0b1100) // 12
    CALL_TX_FP_MUL_FP32(0b1101) // 13
    CALL_TX_FP_MUL_FP32(0b1110) // 14
    CALL_TX_FP_MUL_FP32(0b1111) // 15
  default:
    break;
  }

#undef CALL_TX_FP_ADD_FP32
  __tx_fence();

  __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_max_fp32(void *input0, void *dst, int m, int n,
                                      const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis == 1) // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
    __tx_fp_reduce_max_fp32<aTCSR, cTCSR, 0>((void *)nram0, (void *)nram2);
  if (axis == 1)
    __tx_fp_reduce_max_fp32<aTCSR, cTCSR, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_min_fp32(void *input0, void *dst, int m, int n,
                                      const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis == 1) // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
    __tx_fp_reduce_min_fp32<aTCSR, cTCSR, 0>((void *)nram0, (void *)nram2);
  if (axis == 1)
    __tx_fp_reduce_min_fp32<aTCSR, cTCSR, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}

__device__ void tx_fp_reduce_add_fp32(void *input0, void *dst, int m, int n,
                                      const int axis) {

  configTCSRs(n, m); // m x n;  dim0 = n, dim1 = m
  __tx_async_nram_load<aTCSR, cTCSR, 2>(input0, (void *)nram0);
  if (axis == 0) // reduce on dim0, so the result should be m x 1
    configTCSRs(1, m);
  if (axis == 1) // reduce on dim1, so the result should be 1 x n
    configTCSRs(n, 1);

  __tx_async_nram_load<aTCSR, cTCSR, 2>(dst, (void *)nram2);
  __tx_fence();

  configTCSRs(n, m);
  if (axis == 0)
    __tx_fp_reduce_add_fp32<aTCSR, cTCSR, 0>((void *)nram0, (void *)nram2);
  if (axis == 1)
    __tx_fp_reduce_add_fp32<aTCSR, cTCSR, 1>((void *)nram0, (void *)nram2);

  __tx_fence();

  if (axis == 0)
    configTCSRs(1, m);
  if (axis == 1)
    configTCSRs(n, 1);
  __tx_async_nram_store<aTCSR, cTCSR, 2>((void *)nram2, dst);
  __tx_fence();
}