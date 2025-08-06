#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"

typedef uint8_t fp8;

#define check(x, expect)                                                       \
  if (isnan(x) || (x) != (expect)) {                                           \
    printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, \
           m, n, #x, (x), (expect));                                           \
    exit(-1);                                                                  \
  }

#define fp_check(x, expect, eps)                                               \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {          \
    printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x,     \
           (x), (expect));                                                     \
    exit(-1);                                                                  \
  }

#define fp_checkr(m, n, x, expect, eps)                                        \
  do {                                                                         \
    float _x = (x);                                                            \
    float _expect = (expect);                                                  \
    if (isnan(_x) || (_x) < (_expect) - (eps) || (_x) > (_expect) + (eps)) {   \
      uint32_t x_bits, expect_bits;                                            \
      memcpy(&x_bits, &_x, sizeof(x_bits));                                    \
      memcpy(&expect_bits, &_expect, sizeof(expect_bits));                     \
      printf("Error: %s:%d: (%d,%d) %s = %f (0x%08x), expected %f (0x%08x)\n", \
             __FILE__, __LINE__, m, n, #x, _x, x_bits, _expect, expect_bits);  \
    }                                                                          \
  } while (0)

const int M = 2;
const int K = 3;
const int N = 4;

float a[M * K] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
};

float b[K * N] = {
    1.0f,  2.0f,  3.0f,  4.0f,
    5.0f,  6.0f,  7.0f,  8.0f,
    9.0f, 10.0f, 11.0f, 12.0f
};
/*
A row major 2×3
  [1,2,3]
  [4,5,6]

B row major 3×4
  [1, 2, 3, 4]
  [5, 6, 7, 8]
  [9,10,11,12]
*/
float c0_real[M * N] = {
    38.0f,  44.0f,  50.0f,  56.0f, 
    83.0f,  98.0f, 113.0f, 128.0f
};

/*
A row major 2×3
  [1,2,3]
  [4,5,6]

B col major 3x4
  [1,4,7,10]
  [2,5,8,11]
  [3,6,9,12]
*/
float c1_real[M * N] = {
    14.0, 32.0, 50.0,  68.0,
    32.0, 77.0, 122.0, 167.0,
};

/*
A col major 2×3
  [1,3,5]
  [2,4,6]

B row major 3x4
  [1, 2, 3, 4]
  [5, 6, 7, 8]
  [9,10,11,12]
*/
float c2_real[M * N] = {
    61.0, 70.0, 79.0, 88.0,
    76.0, 88.0, 100.0, 112.0
};

/*
A col major 2×3
  [1,3,5]
  [2,4,6]

B col major 3x4
  [1,4,7,10]
  [2,5,8,11]
  [3,6,9,12]
*/
float c3_real[M * N] = {
  22.0, 49.0, 76.0, 103.0,
  28.0, 64.0, 100.0, 136.0,
};

fp8 a_e4m3[M * K];
fp8 b_e4m3[K * N];

float c0[M * N];
float c1[M * N];
float c2[M * N];
float c3[M * N];

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_0_addr = 0x81000000;
static uint64_t src_1_addr = 0x82000000;
static uint64_t dst_0_addr = 0x83000000;

static fp8 __f32_to_e4m3(float f) { return RVT_EMU_FP8::from_float<4>(f); }

static float __e4m3_to_f32(fp8 e4m3) {
  RVT_EMU_FP8::FP8<4> data;
  data.bits = e4m3;
  return RVT_EMU_FP8::to_float(data);
}

static void trans_f32_to_e4m3() {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      a_e4m3[m * K + k] = __f32_to_e4m3(a[m * K + k]);
      fp_check(__e4m3_to_f32(a_e4m3[m * K + k]), a[m * K + k], 0.0f);
    }
  }
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      b_e4m3[k * N + n] = __f32_to_e4m3(b[k * N + n]);
      fp_check(__e4m3_to_f32(b_e4m3[k * N + n]), b[k * N + n], 0.0f);
    }
  }
}

static void testrt_npu_init() {
  ssize_t err = 0;
  printf("[INFO] Initializing Runtime..\n");
  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to initialize Runtime: err=%ld\n", err);
    exit(-1);
  }
  printf("[INFO] Runtime initialized successfully.\n");
}

static void testrt_npu_load_kernel_set() {
  ssize_t err = 0;
  err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
  if (err != NPU_SUCCESS) {
    printf("Failed to load kernel set: err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Kernel set loaded successfully.\n");
}

static void testrt_run_tx_init_kernel() {
  ssize_t err = 0;
  struct NPUKernelDesc *tx_init_kernel_desc;
  err =
      npu_runtime_get_kernel_desc(&runtime, "__tx_init", &tx_init_kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description for __tx_init: err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded __tx_init kernel desc successfully.\n");
  int kernel_ret;
  printf("[INFO] Start kernel __tx_init ... \n");
  err = npu_runtime_kernel_start(&runtime, tx_init_kernel_desc, &kernel_ret, 0);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Kernel __tx_init run successfully.\n");
}

static void testrt_npu_get_kernel_desc(const char *kernel_name) {
  ssize_t err = 0;
  err = npu_runtime_get_kernel_desc(&runtime, kernel_name, &kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description for '%s': err=%ld\n", kernel_name,
           err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded '%s' kernel desc successfully.\n", kernel_name);
}

void check_result(float *c, float *c_real) {
  printf("[INFO] Check result ...\n");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      fp_checkr(m, n, c[m * N + n], c_real[m * N + n], 0.1f);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

static void testrt_npu_memcpy_in(void *input0, void *input1, void *output) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)input0,
                          M * K * sizeof(fp8));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n",
           src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         M * K * sizeof(fp8), src_0_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, src_1_addr, (void *)input1,
                          K * N * sizeof(fp8));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n",
           src_1_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         K * N * sizeof(fp8), src_1_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)output,
                          M * N * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         M * N * sizeof(float), dst_0_addr);
}

static void testrt_npu_memcpy_out(void *output) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst_0_addr,
                            M * N * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
         M * N * sizeof(int32_t), dst_0_addr);
}

int run_kenrel(const char *kernel_name, void *input0, void *input1,
               void *output) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  printf("[INFO] M=%d, N=%d\n", M, N);

  // Copy in
  testrt_npu_memcpy_in(input0, input1, output);

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel %s ... \n", kernel_name);
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 6,
                                 src_0_addr, src_1_addr, dst_0_addr, M, N, K);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output);
  return 0;
}

int main(int argc, char **argv) {

  trans_f32_to_e4m3();

  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  run_kenrel("matmul_e4m3_e4m3_fp32_row_row", a_e4m3, b_e4m3, c0);
  check_result(c0, c0_real);

  run_kenrel("matmul_e4m3_e4m3_fp32_row_col", a_e4m3, b_e4m3, c1);
  check_result(c1, c1_real);

  run_kenrel("matmul_e4m3_e4m3_fp32_col_row", a_e4m3, b_e4m3, c2);
  check_result(c2, c2_real);

  run_kenrel("matmul_e4m3_e4m3_fp32_col_col", a_e4m3, b_e4m3, c3);
  check_result(c3, c3_real);

  // End
  npu_runtime_deinit(&runtime);

  // Check result

  return 0;
}
