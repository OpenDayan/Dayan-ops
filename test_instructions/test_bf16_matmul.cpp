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

const long M = 128, N = 512, K = 512;
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

int main(int argc, char** argv) {
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

  // 初始化计算用矩阵 A = m, n == m * 128 + n
  for(int m = 0 ; m < M; m ++) {
    for(int k = 0 ; k < K; k++) {
      // A[m][k] = FP32ToBF16((((double)rand() / (double)RAND_MAX) - 0.5f) * 8.f);
      // A[m][k] = FP32ToBF16((float)(m * K + k));
      A[m][k] = m * K + k;
    }
  }

  // 初始化计算用矩阵 B = I
  for(int n = 0 ; n < N; n ++) {
    for(int k = 0 ; k < K; k++) {
      if(n == k) B[n][k] = FP32ToBF16(1.f);
      else B[n][k] = FP32ToBF16(0.f);
    }
  }

  // 分配 device 空间
  uint64_t dev_a, dev_b, dev_c;
  npu_runtime_malloc(&runtime, &dev_a, M * K * 2);
  npu_runtime_malloc(&runtime, &dev_b, N * K * 2);
  npu_runtime_malloc(&runtime, &dev_c, M * N * 4);

  std::cout << "dev_a: " << std::hex << dev_a << " dev_b: " << dev_b << " dev_c: " << dev_c << std::endl;

  npu_memcpy_to_dev(&runtime.dev_state, dev_a, A, M * K * 2);
  npu_memcpy_to_dev(&runtime.dev_state, dev_b, B, N * K * 2);
  npu_memcpy_to_dev(&runtime.dev_state, dev_c, C, M * N * 4);

  auto test_name = "matmul_bf16_a_bt_c";

  struct NPUKernelDesc *kernel_desc;
  err = npu_runtime_get_kernel_desc(&runtime, test_name, &kernel_desc);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to get kernel description: " << test_name << std::endl;
    return -1;
  }
  int kernel_ret;
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 9,
    M, N, K,
    dev_a, dev_b, dev_c,
    K * 2, K * 2, N * 4
  );

  std::cout << "All Instr Test Finished." << std::endl;

  npu_memcpy_from_dev(&runtime.dev_state, C, dev_c, M * N * 4);

  FILE* r = fopen("ref.rpt", "w");
  FILE* f = fopen("output.rpt", "w");

  for(int m = 0; m < M; m++) {
    for(int n = 0; n < N; n++) {
      auto diff = FP32ToBF16(C[m][n]) - A[m][n];
      if(diff) {
        std::cout << std::hex << "Diff @" << m << ", " << n << ": ref=" << A[m][n] << " dut=" << FP32ToBF16(C[m][n]) << std::endl;
        // return -1;
      }
      fprintf(r, "%04x ", A[m][n]);
      fprintf(f, "%04x ", FP32ToBF16(C[m][n]));
    }
    fprintf(r, "\n");
    fprintf(f, "\n");
  }

  npu_runtime_deinit(&runtime);

  return 0;
}
