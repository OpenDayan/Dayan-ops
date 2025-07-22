#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <float.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"

#define check(x, expect)                                                       \
  if ((x) != (expect)) {                                                       \
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

#include "generated/sfu_size.gen"

float a_fp32_in[M * N] = {
#include "generated/sfu_float_a_row.gen"
};

float a_recip_real[M * N] = {
#include "generated/sfu_a_recip_row.gen"
};

float a_rsqrt_real[M * N] = {
#include "generated/sfu_a_rsqrt_row.gen"
};

float a_sin_real[M * N] = {
#include "generated/sfu_a_sin_row.gen"
};

float a_cos_real[M * N] = {
#include "generated/sfu_a_cos_row.gen"
};

float a_exp_real[M * N] = {
#include "generated/sfu_a_exp_row.gen"
};

float c[M * N] = {0};

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_addr = 0x81000000;
static uint64_t dst_addr = 0x82000000;
static uint64_t tcsr_addr = 0x84000000;

static TCFG tcsr[2];

void init_tcfg(const long dims[2][4], const long strides[2][4]) {
  for (int i = 0; i < 2; ++i) {
    tcsr[i] = {.dim0 = dims[i][0],
               .stride0 = strides[i][0],
               .dim1 = dims[i][1],
               .stride1 = strides[i][1],
               .dim2 = dims[i][2],
               .stride2 = strides[i][2],
               .dim3 = dims[i][3],
               .stride3 = strides[i][3]};
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

static void testrt_npu_memcpy_in(void *input, void *output,
                                 const long dims[2][4],
                                 uint32_t input_type_size,
                                 uint32_t output_type_size) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  size_t input_size = dims[0][0] * dims[0][1] * dims[0][2] * dims[0][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_addr, (void *)input,
                          input_size * sizeof(input_type_size));
  if (ret < 0)
    printf("[ERROR] Failed to write src to device at 0x%lx: err=%ld\n",
           src_addr, ret);
  printf("[INFO] Copied src(%ld Bytes) to device at 0x%lx\n",
         input_size * sizeof(input_type_size), src_addr);

  size_t output_size = dims[1][0] * dims[1][1] * dims[1][2] * dims[1][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_addr, (void *)output,
                          output_size * sizeof(output_type_size));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         output_size * sizeof(output_type_size), dst_addr);
}

static void testrt_npu_memcpy_out(void *output, uint32_t output_type_size) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst_addr,
                            M * N * output_type_size);
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_addr, ret);
  printf("[INFO] Copied dst0(%d Bytes) from device at 0x%lx\n",
         M * N * output_type_size, dst_addr);
}

void check_fp32_result(float *output_real) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      // printf("c[] is %f\n", c[n * M + m]);
      fp_check(c[n * M + m], output_real[n * M + m], 0.001f);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

void check_fp32_exp_result(float *output_real) {
  printf("[INFO] Check result ...\n");
  int error_count = 0;
  const float relative_tolerance = 1e-5f;
  const float absolute_tolerance = 1.0f;

  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      float expected = c[n * M + m];
      float actual = output_real[n * M + m];

      float abs_error = fabsf(expected - actual);
      float rel_error = abs_error / fmaxf(fabsf(expected), FLT_MIN);

      if (abs_error > absolute_tolerance && rel_error > relative_tolerance) {
        printf("[ERROR] At [%d][%d]: expected %.6f, got %.6f (abs_err=%.2e, "
               "rel_err=%.2e%%)\n",
               n, m, expected, actual, abs_error, rel_error * 100.0f);
        error_count++;
      }
    }
  }

  if (error_count == 0) {
    printf("[INFO] All results within tolerance!\n");
  } else {
    printf("[WARN] %d/%d results exceeded tolerance\n", error_count, N * M);
  }
}

void testrt_npu_config_tcsrs(const long dims[2][4], const long strides[2][4]) {
  int err = 0;

  init_tcfg(dims, strides);
  err = npu_memcpy_to_dev(&runtime.dev_state, tcsr_addr, (void *)tcsr,
                          sizeof(tcsr));
  printf("[INFO] Copy tcsr to device with size:%ld and ret:%d\n", sizeof(tcsr),
         err);
}

int run_kernel(const char *kernel_name, void *input, void *output,
               const long dims[2][4], const long strides[2][4],
               uint32_t input_type_size, uint32_t output_type_size) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  // Copy in
  testrt_npu_memcpy_in(input, output, dims, input_type_size, output_type_size);
  testrt_npu_config_tcsrs(dims, strides);

  printf("[INFO] Start kernel %s ... \n", kernel_name);

  int kernel_ret = 0;
  int err = 0;
  npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 3, src_addr,
                           dst_addr, tcsr_addr);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%d\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output, output_type_size);
  return 0;
}

int main(int argc, char **argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  const long dims[2][4] = {{N, M, 1, 1}, {N, M, 1, 1}};
  const long strides[2][4] = {{1, N, M * N, M * N}, {1, N, M * N, M * N}};

  run_kernel("tx_sfu_recip_fp32", a_fp32_in, c, dims, strides, 4, 4);
  check_fp32_result(a_recip_real);

  run_kernel("tx_sfu_rsqrt_fp32", a_fp32_in, c, dims, strides, 4, 4);
  check_fp32_result(a_rsqrt_real);

  run_kernel("tx_sfu_sin_fp32", a_fp32_in, c, dims, strides, 4, 4);
  check_fp32_result(a_sin_real);

  run_kernel("tx_sfu_cos_fp32", a_fp32_in, c, dims, strides, 4, 4);
  check_fp32_result(a_cos_real);

  run_kernel("tx_sfu_exp_fp32", a_fp32_in, c, dims, strides, 4, 4);
  check_fp32_exp_result(a_exp_real);

  npu_runtime_deinit(&runtime);
  return 0;
}
