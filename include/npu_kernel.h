#ifndef _NPU_KERNEL_H_
#define _NPU_KERNEL_H_

#ifdef __cplusplus
#define __device__ extern "C" __attribute__((section(".text.npu_kernel")))
#else
#define __device__ __attribute__((section(".text.npu_kernel")))
#endif

#endif
