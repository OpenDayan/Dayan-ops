#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"

extern struct NPUDeviceKernelSetDesc __bin_kernels_pack;

const long M = 1536, N = 256, K = 1536;
uint16_t A[M][K];
uint16_t B[N][K];
float C[M][N];

union fp32_int32 {
    float f;
    uint32_t bits;
};

static inline uint16_t FP32ToBF16(float a) {
    fp32_int32 u;
    u.f = a;
    return (u.bits >> 16) & 0xffff;
}

static inline float BF16ToFP32(uint16_t a) {
    fp32_int32 u;
    u.bits = a << 16;
    return u.f;
}

int main(int argc, char **argv) {
    std::cout << "Initializing Runtime..." << std::endl;
    struct NPURuntimeState runtime = {0};
    ssize_t err;
    err = npu_runtime_init(&runtime);
    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to initialize Runtime: " << err << std::endl;
        return -1;
    }
    std::cout << "Runtime initialized successfully." << std::endl;

    err = npu_runtime_load_kernelset(&runtime, &__bin_kernels_pack);
    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to load kernel set: " << err << std::endl;
        npu_runtime_deinit(&runtime);
        return -1;
    }

    std::cout << "Kernel set loaded successfully." << std::endl;

    // 分配 device 空间
    uint64_t dev_a, dev_b, dev_c;
    npu_runtime_malloc(&runtime, &dev_a, M * K * 2);
    npu_runtime_malloc(&runtime, &dev_b, N * K * 4);
    npu_runtime_malloc(&runtime, &dev_c, M * N * 4);

    std::cout << "dev_a: " << std::hex << dev_a << " dev_b: " << dev_b << " dev_c: " << dev_c
              << std::endl;

    auto test_name = "gmma_bf16_fp32_AT_QK256";

    struct NPUKernelDesc *kernel_desc;
    err = npu_runtime_get_kernel_desc(&runtime, test_name, &kernel_desc);
    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to get kernel description: " << test_name << std::endl;
        return -1;
    }
    int kernel_ret;
    err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 14,
        dev_a, dev_b, dev_c,
        M, N, K,
        K * 2,
        K * 4,
        N * 4,
        1, 1, 0, 0, 0);

    std::cout << "All Instr Test Finished." << std::endl;

    npu_runtime_deinit(&runtime);

    return 0;
}
