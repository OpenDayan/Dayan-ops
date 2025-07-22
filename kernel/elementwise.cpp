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

    if (a_dim0 != b_dim0 && (a_dim0 == 1 || b_dim0 == 1)) {
        broadcast |= 0b0001;
    }

    if (a_dim1 != b_dim1 && (a_dim1 == 1 || b_dim1 == 1)) {
        broadcast |= 0b0010;
    }

    if (a_dim2 != b_dim2 && (a_dim2 == 1 || b_dim2 == 1)) {
        broadcast |= 0b0100;
    }

    if (a_dim3 != b_dim3 && (a_dim3 == 1 || b_dim3 == 1)) {
        broadcast |= 0b1000;

    }

    return broadcast;
}

__device__ long kernel_add_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{

    // 计算总元素数
    long total_elements = (long)psrc0_ne3 * psrc0_ne2 * psrc0_ne1 * psrc0_ne0;
    long b_total_elements = (long)psrc1_ne3 * psrc1_ne2 * psrc1_ne1 * psrc1_ne0;

// NRAM地址分配 - 256KB = 262144 bytes，每个float占4字节
    // 每个tensor分配大约80KB空间，留一些buffer
    const uint64_t nram0 = 0x0;        // A tensor: 0x0 - 0x14000 (80KB)
    const uint64_t nram1 = 0x14000;    // B tensor: 0x14000 - 0x28000 (80KB)  
    const uint64_t nram2 = 0x28000;    // C tensor: 0x28000 - 0x3C000 (80KB)
    // 剩余空间: 0x3C000 - 0x40000 (16KB) 作为buffer

    // TCFG配置
    const int aTCSR = 0;
    const int bTCSR = 1;
    const int cTCSR = 2;

    // 计算广播参数
    // uint8_t broadcast = get_broadcast(psrc0_ne0, psrc0_ne1, psrc0_ne2, psrc0_ne3,
    //                                  psrc1_ne0, psrc1_ne1, psrc1_ne2, psrc1_ne3);
    uint8_t broadcast = 0b0000;

    config_tcsr_simple<aTCSR>();
    config_tcsr_simple<bTCSR>();
    config_tcsr_simple<cTCSR>();

    // 计算各维度的stride (类似CPU版本)
    int a_stride_dim0 = 1;
    int a_stride_dim1 = psrc0_ne0;
    int a_stride_dim2 = psrc0_ne0 * psrc0_ne1;
    int a_stride_dim3 = psrc0_ne0 * psrc0_ne1 * psrc0_ne2;
    
    int b_stride_dim0 = 1;
    int b_stride_dim1 = psrc1_ne0;
    int b_stride_dim2 = psrc1_ne0 * psrc1_ne1;
    int b_stride_dim3 = psrc1_ne0 * psrc1_ne1 * psrc1_ne2;

    // 4D循环遍历 (参考CPU版本的循环结构)
    for (int i = 0; i < psrc0_ne3; ++i) {
        // 处理广播：如果psrc1在该维度为1，则使用索引0
        int i1 = (psrc1_ne3 == 1) ? 0 : i;
        
        for (int j = 0; j < psrc0_ne2; ++j) {
            int j1 = (psrc1_ne2 == 1) ? 0 : j;
            
            for (int k = 0; k < psrc0_ne1; ++k) {
                int k1 = (psrc1_ne1 == 1) ? 0 : k;
                
                for (int p = 0; p < psrc0_ne0; ++p) {
                    int p1 = (psrc1_ne0 == 1) ? 0 : p;
                    
                    // 计算线性偏移 (参考CPU版本的地址计算)
                    long a_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + p * a_stride_dim0;
                    long b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + p1 * b_stride_dim0;
                    long c_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + p * a_stride_dim0;  // C与A相同形状

                    // 计算指针位置
                    float* a_elem_ptr = (float*)a_ptr + a_offset;
                    float* b_elem_ptr = (float*)b_ptr + b_offset;
                    float* c_elem_ptr = (float*)c_ptr + c_offset;

                    // 加载数据到NRAM
                    __tx_async_nram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)nram0);
                    __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
                    __tx_fence();

                    // 执行向量加法运算
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

                    // 存储结果回内存
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, c_elem_ptr);
                    __tx_fence();
                }
            }
        }
    }

    printf("[KERNEL] NPU kernel execution completed successfully!\n");
    return 0;
}

__device__ long kernel_mul_fp32(void *a_ptr, void *b_ptr, void *c_ptr,
                                int psrc0_ne0, int psrc0_ne1, int psrc0_ne2, int psrc0_ne3,
                                int psrc1_ne0, int psrc1_ne1, int psrc1_ne2, int psrc1_ne3)
{
    // 计算总元素数
    long total_elements = (long)psrc0_ne3 * psrc0_ne2 * psrc0_ne1 * psrc0_ne0;
    long b_total_elements = (long)psrc1_ne3 * psrc1_ne2 * psrc1_ne1 * psrc1_ne0;

    // NRAM地址分配 - 256KB = 262144 bytes，每个float占4字节
    // 每个tensor分配大约80KB空间，留一些buffer
    const uint64_t nram0 = 0x0;        // A tensor: 0x0 - 0x14000 (80KB)
    const uint64_t nram1 = 0x14000;    // B tensor: 0x14000 - 0x28000 (80KB)  
    const uint64_t nram2 = 0x28000;    // C tensor: 0x28000 - 0x3C000 (80KB)
    // 剩余空间: 0x3C000 - 0x40000 (16KB) 作为buffer

    // TCFG配置
    const int aTCSR = 0;
    const int bTCSR = 1;
    const int cTCSR = 2;

    // 由于逐元素处理，不需要NPU层面的广播
    uint8_t broadcast = 0b0000;

    config_tcsr_simple<aTCSR>();
    config_tcsr_simple<bTCSR>();
    config_tcsr_simple<cTCSR>();

    // 计算各维度的stride (类似CPU版本)
    int a_stride_dim0 = 1;
    int a_stride_dim1 = psrc0_ne0;
    int a_stride_dim2 = psrc0_ne0 * psrc0_ne1;
    int a_stride_dim3 = psrc0_ne0 * psrc0_ne1 * psrc0_ne2;
    
    int b_stride_dim0 = 1;
    int b_stride_dim1 = psrc1_ne0;
    int b_stride_dim2 = psrc1_ne0 * psrc1_ne1;
    int b_stride_dim3 = psrc1_ne0 * psrc1_ne1 * psrc1_ne2;

    // 4D循环遍历 (参考CPU版本的循环结构)
    for (int i = 0; i < psrc0_ne3; ++i) {
        // 处理广播：如果psrc1在该维度为1，则使用索引0
        int i1 = (psrc1_ne3 == 1) ? 0 : i;
        
        for (int j = 0; j < psrc0_ne2; ++j) {
            int j1 = (psrc1_ne2 == 1) ? 0 : j;
            
            for (int k = 0; k < psrc0_ne1; ++k) {
                int k1 = (psrc1_ne1 == 1) ? 0 : k;
                
                for (int p = 0; p < psrc0_ne0; ++p) {
                    int p1 = (psrc1_ne0 == 1) ? 0 : p;
                    
                    // 计算线性偏移 (参考CPU版本的地址计算)
                    long a_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + p * a_stride_dim0;
                    long b_offset = i1 * b_stride_dim3 + j1 * b_stride_dim2 + k1 * b_stride_dim1 + p1 * b_stride_dim0;
                    long c_offset = i * a_stride_dim3 + j * a_stride_dim2 + k * a_stride_dim1 + p * a_stride_dim0;  // C与A相同形状

                    // 计算指针位置
                    float* a_elem_ptr = (float*)a_ptr + a_offset;
                    float* b_elem_ptr = (float*)b_ptr + b_offset;
                    float* c_elem_ptr = (float*)c_ptr + c_offset;

                    // 加载数据到NRAM
                    __tx_async_nram_load<aTCSR, aTCSR, 2>(a_elem_ptr, (void *)nram0);
                    __tx_async_nram_load<bTCSR, bTCSR, 2>(b_elem_ptr, (void *)nram1);
                    __tx_fence();

                    // 执行向量乘法运算
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
                        printf("[KERNEL] ERROR: Unsupported broadcast pattern: %d\n", broadcast);
                        return -1;
                    }

#undef CALL_TX_FP_MUL_FP32

                    __tx_fence();

                    // 存储结果回内存
                    __tx_async_nram_store<cTCSR, cTCSR, 2>((void *)nram2, c_elem_ptr);
                    __tx_fence();
                }
            }
        }
    }

    printf("[KERNEL] NPU kernel_mul_fp32 execution completed successfully!\n");
    return 0;
}