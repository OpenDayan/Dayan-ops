#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>


template<int TCSR_ID>
static inline void config_tcsr_simple() {
    
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE2>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE3>(1);
}

// 广播计算
static inline uint8_t get_broadcast(int a_dim0, int a_dim1, int a_dim2, int a_dim3,
                                   int b_dim0, int b_dim1, int b_dim2, int b_dim3) {
    uint8_t broadcast = 0;

    printf("[KERNEL] Broadcast calculation:\n");
    printf("[KERNEL]   dim0: %d vs %d", a_dim0, b_dim0);
    if (a_dim0 != b_dim0 && (a_dim0 == 1 || b_dim0 == 1)) {
        broadcast |= 0b0001;
        printf(" -> BROADCAST");
    }
    printf("\n");

    printf("[KERNEL]   dim1: %d vs %d", a_dim1, b_dim1);
    if (a_dim1 != b_dim1 && (a_dim1 == 1 || b_dim1 == 1)) {
        broadcast |= 0b0010;
        printf(" -> BROADCAST");
    }
    printf("\n");

    printf("[KERNEL]   dim2: %d vs %d", a_dim2, b_dim2);
    if (a_dim2 != b_dim2 && (a_dim2 == 1 || b_dim2 == 1)) {
        broadcast |= 0b0100;
        printf(" -> BROADCAST");
    }
    printf("\n");

    printf("[KERNEL]   dim3: %d vs %d", a_dim3, b_dim3);
    if (a_dim3 != b_dim3 && (a_dim3 == 1 || b_dim3 == 1)) {
        broadcast |= 0b1000;
        printf(" -> BROADCAST");
    }
    printf("\n");

    printf("[KERNEL] Final broadcast mask: 0b");
    if (broadcast & 0b1000) printf("1"); else printf("0");
    if (broadcast & 0b0100) printf("1"); else printf("0");
    if (broadcast & 0b0010) printf("1"); else printf("0");
    if (broadcast & 0b0001) printf("1"); else printf("0");
    printf(" (%d)\n", broadcast);

    return broadcast;
}

__device__ long kernel_add_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{
    long total_elements = (long)psrc0_ne3 * psrc0_ne2 * psrc0_ne1 * psrc0_ne0;
    long b_total_elements = (long)psrc1_ne3 * psrc1_ne2 * psrc1_ne1 * psrc1_ne0;

    const uint64_t nram0 = 0x1000000;  // A tensor
    const uint64_t nram1 = 0x2000000;  // B tensor  
    const uint64_t nram2 = 0x3000000;  // C tensor (result)

    // TCFG配置
    const int aTCSR = 0;
    const int bTCSR = 1;
    const int cTCSR = 2;

    // 计算广播参数
    uint8_t broadcast = get_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
                                     psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);

    config_tcsr_simple<aTCSR>();
    config_tcsr_simple<bTCSR>();
    config_tcsr_simple<cTCSR>();

    for (long elem_idx = 0; elem_idx < total_elements; elem_idx++) {
        // 计算当前元素的内存偏移
        float* a_elem_ptr = (float*)a_ptr + elem_idx;
        float* c_elem_ptr = (float*)c_ptr + elem_idx;
        
        long b_elem_idx = elem_idx;
        if (b_total_elements < total_elements) {
            b_elem_idx = elem_idx % b_total_elements;
        }
        float* b_elem_ptr = (float*)b_ptr + b_elem_idx;

        __tx_async_nram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)nram0);
        __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
        __tx_fence();

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
                  printf("[KERNEL] ERROR: Unsupported broadcast pattern: %d\n", broadcast);
                  return -1;
              }
      
      #undef CALL_TX_FP_ADD_FP32

        __tx_fence();

        __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, c_elem_ptr);
        __tx_fence();
    }

    // printf("[KERNEL] c_ptr address: %p, c[0]=%x\n", c_ptr, ((float*)c_ptr)[0]);
    // printf("[KERNEL] a_ptr address: %p, a[0]=%x\n", a_ptr, ((float*)a_ptr)[0]);
    // printf("[KERNEL] b_ptr address: %p, b[0]=%x\n", b_ptr, ((float*)b_ptr)[0]);
    // float a_val = ((float*)a_ptr)[0];
    // float b_val = ((float*)b_ptr)[0];
    // ((float*)c_ptr)[0] = a_val + b_val;
    // float c_val = ((float*)c_ptr)[0];
    // printf("[KERNEL] a[0]=%f, b[0]=%f, c[0]=%f\n", a_val, b_val, c_val);

    printf("[KERNEL] NPU kernel execution completed successfully!\n");
    return 0;
}