#ifndef _NPU_KERNEL_H_
#define _NPU_KERNEL_H_

#include <stdint.h>

#ifdef __cplusplus
#define __device__ extern "C" __attribute__((section(".text.npu_kernel")))
#else
#define __device__ __attribute__((section(".text.npu_kernel")))
#endif

__attribute__((always_inline)) static inline uint32_t float_as_u32(float i) { return *(uint32_t*)&i; }
__attribute__((always_inline)) static inline float u32_as_float(uint32_t i) { return *(float*)&i; }

#endif
