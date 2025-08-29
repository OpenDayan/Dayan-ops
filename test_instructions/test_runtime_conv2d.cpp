#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"

typedef uint16_t fp16;

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

#define BATCH 1
#define IN_H 3
#define IN_W 3
#define IN_C 1
#define FILTER_H 2
#define FILTER_W 2
#define OUT_C 1
#define PAD_H 0
#define PAD_W 0
#define STRIDE_H 1
#define STRIDE_W 1
#define OUT_H 2
#define OUT_W 2

float a_conv_fp32[BATCH * IN_H * IN_W * IN_C] = {
    // input 3x3x1
    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

float b_conv_fp32[OUT_C * FILTER_H * FILTER_W * IN_C] = {
    // kernel 2x2x1
    1.0f, 1.0f, 1.0f, 1.0f};

float c_conv_real_fp32[BATCH * OUT_H * OUT_W * OUT_C] = {
    // output 2x2x1
    12.0f, 16.0f, 24.0f, 28.0f};

float c_output[BATCH * OUT_H * OUT_W * OUT_C] = {0};

fp16 a_conv_fp16[BATCH * IN_H * IN_W * IN_C];
fp16 b_conv_fp16[OUT_C * FILTER_H * FILTER_W * IN_C];
fp16 c_conv_real_fp16[BATCH * OUT_H * OUT_W * OUT_C];
float c_conv_real_fp32_from_fp16[BATCH * OUT_H * OUT_W * OUT_C];

/* FP16 conversion */
static inline float __fp16_to_fp32(fp16 h) {
  uint32_t x = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) |
               ((h & 0x03ff) << 13);
  if ((h & 0x7c00) == 0) {
    if ((h & 0x03ff) != 0) {
      uint32_t sign = (h & 0x8000) << 16;
      uint32_t mantissa = (h & 0x03ff);
      int32_t exponent = -14;

      while ((mantissa & 0x0400) == 0) {
        mantissa <<= 1;
        exponent--;
      }

      mantissa &= 0x03ff;
      x = sign | ((exponent + 127) << 23) | (mantissa << 13);
    }
  }

  float r = 0.0f;
  memcpy(&r, &x, sizeof(uint32_t));
  return r;
}

static inline fp16 __fp32_to_fp16(float f) {
  uint32_t x = 0;
  memcpy(&x, &f, sizeof(float));
  fp16 h = ((x >> 16) & 0x8000) |
           ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
           ((x >> 13) & 0x03ff);

  if ((x & 0x7f800000) == 0x7f800000) {
    h = (h & 0x8000) | 0x7c00 | ((x >> 13) & 0x03ff);
  } else if ((x & 0x7f800000) > 0x477fe000) {
    h = (h & 0x8000) | 0x7c00;
  } else if ((x & 0x7f800000) < 0x38000000) {
    uint32_t sign = x & 0x80000000;
    uint32_t mantissa = (x & 0x007fffff) | 0x00800000;
    int32_t exponent = ((x >> 23) & 0xff) - 127 - 14;

    if (exponent < -24) {
      h = (fp16)(sign >> 16);
    } else {
      mantissa >>= (-exponent);
      h = (fp16)((sign >> 16) | (mantissa >> 13));
    }
  }
  return h;
}

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src0_addr = 0x81000000;
static uint64_t src1_addr = 0x82000000;
static uint64_t dst0_addr = 0x83000000;
static uint64_t tcsr_addr = 0x84000000;

static TCFG tcsr[4];

void init_tcfg(const long dims[3][4], const long strides[3][4],
               const long conv_info[4]) {
  for (int i = 0; i < 3; ++i) {
    tcsr[i] = {.dim0 = dims[i][0],
               .stride0 = strides[i][0],
               .dim1 = dims[i][1],
               .stride1 = strides[i][1],
               .dim2 = dims[i][2],
               .stride2 = strides[i][2],
               .dim3 = dims[i][3],
               .stride3 = strides[i][3]};
  }
  tcsr[3] = {
      .dim0 = conv_info[0],
      .stride0 = conv_info[1],
      .dim1 = conv_info[2],
      .stride1 = conv_info[3],
  };
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

static void testrt_npu_memcpy_in(void *input0, void *input1, void *output,
                                 const long dims[3][4], size_t input_elem_size,
                                 size_t output_elem_size) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  size_t input0_size = dims[0][0] * dims[0][1] * dims[0][2] * dims[0][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, src0_addr, (void *)input0,
                          input0_size * sizeof(input_elem_size));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n",
           src0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         input0_size * sizeof(input_elem_size), src0_addr);

  size_t input1_size = dims[1][0] * dims[1][1] * dims[1][2] * dims[1][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, src1_addr, (void *)input1,
                          input1_size * sizeof(input_elem_size));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n",
           src1_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         input1_size * sizeof(input_elem_size), src1_addr);

  size_t output_size = dims[2][0] * dims[2][1] * dims[2][2] * dims[2][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, dst0_addr, (void *)output,
                          output_size * sizeof(output_elem_size));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         output_size * sizeof(output_elem_size), dst0_addr);
}

static void testrt_npu_memcpy_out(void *output, size_t output_size,
                                  size_t output_elem_size) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst0_addr,
                            output_size * output_elem_size);
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
         output_size * output_elem_size, dst0_addr);
}

void check_conv_result(float *output_real, float *expected, int batch,
                       int height, int width, int channels) {
  printf("[INFO] Check conv result ...\n");
  int total_elements = batch * height * width * channels;
  for (int i = 0; i < total_elements; i++) {
    fp_check(output_real[i], expected[i], 0.001f);
  }
  printf("[INFO] Conv result check successfully!\n");
}

void testrt_npu_config_tcsrs(const long dims[3][4], const long strides[3][4]) {
  int err = 0;
  const long convInfo[4] = {PAD_H, PAD_W, STRIDE_H, STRIDE_W};
  init_tcfg(dims, strides, convInfo);
  err = npu_memcpy_to_dev(&runtime.dev_state, tcsr_addr, (void *)tcsr,
                          sizeof(tcsr));
  printf("[INFO] Copy tcsr to device with size:%ld and ret:%d\n", sizeof(tcsr),
         err);
}

int run_kernel(const char *kernel_name, void *input0, void *input1,
               void *output, const long dims[3][4], const long strides[3][4],
               size_t input_elem_size, size_t output_elem_size) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  // Copy in
  testrt_npu_memcpy_in(input0, input1, output, dims, input_elem_size,
                       output_elem_size);
  testrt_npu_config_tcsrs(dims, strides);

  printf("[INFO] Start kernel %s ... \n", kernel_name);

  int kernel_ret = 0;
  int err = 0;
  npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 4, src0_addr,
                           src1_addr, dst0_addr, tcsr_addr);

  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%d\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  size_t output_size = dims[2][0] * dims[2][1] * dims[2][2] * dims[2][3];
  testrt_npu_memcpy_out(output, output_size, output_elem_size);
  return 0;
}

void test_conv2d_fp32_fp32_fp32() {

  float c_conv[BATCH * OUT_H * OUT_W * OUT_C] = {0};

  const long dims_conv[3][4] = {
      {BATCH, IN_H, IN_W, IN_C}, // input: [batch, height, width, channels]
      {OUT_C, FILTER_H, FILTER_W,
       IN_C}, // filter: [out_channels, height, width, in_channels]
      {BATCH, OUT_H, OUT_W,
       OUT_C} // output: [batch, height, width, out_channels]
  };

  const long strides_conv[3][4] = {
      {IN_H * IN_W * IN_C, IN_W * IN_C, IN_C, 1},             // input strides
      {FILTER_H * FILTER_W * IN_C, FILTER_W * IN_C, IN_C, 1}, // filter strides
      {OUT_H * OUT_W * OUT_C, OUT_W * OUT_C, OUT_C, 1}        // output strides
  };

  run_kernel("tx_conv_float_fp32_fp32_fp32", a_conv_fp32, b_conv_fp32, c_conv,
             dims_conv, strides_conv, sizeof(float), sizeof(float));

  check_conv_result(c_conv, c_conv_real_fp32, BATCH, OUT_H, OUT_W, OUT_C);
}

void test_conv2d_fp16_fp16_fp16() {

  for (int i = 0; i < BATCH * IN_H * IN_W * IN_C; i++) {
    a_conv_fp16[i] = __fp32_to_fp16(a_conv_fp32[i]);
  }
  for (int i = 0; i < OUT_C * FILTER_H * FILTER_W * IN_C; i++) {
    b_conv_fp16[i] = __fp32_to_fp16(b_conv_fp32[i]);
  }
  for (int i = 0; i < BATCH * OUT_H * OUT_W * OUT_C; i++) {
    c_conv_real_fp16[i] = __fp32_to_fp16(c_conv_real_fp32[i]);
  }

  fp16 c_conv_fp16[BATCH * OUT_H * OUT_W * OUT_C] = {0};

  const long dims_conv[3][4] = {
      {BATCH, IN_H, IN_W, IN_C}, // input: [batch, height, width, channels]
      {OUT_C, FILTER_H, FILTER_W,
       IN_C}, // filter: [out_channels, height, width, in_channels]
      {BATCH, OUT_H, OUT_W,
       OUT_C} // output: [batch, height, width, out_channels]
  };

  const long strides_conv[3][4] = {
      {IN_H * IN_W * IN_C, IN_W * IN_C, IN_C, 1},             // input strides
      {FILTER_H * FILTER_W * IN_C, FILTER_W * IN_C, IN_C, 1}, // filter strides
      {OUT_H * OUT_W * OUT_C, OUT_W * OUT_C, OUT_C, 1}        // output strides
  };

  run_kernel("tx_conv_float_fp16_fp16_fp16", a_conv_fp16, b_conv_fp16,
             c_conv_fp16, dims_conv, strides_conv, sizeof(fp16), sizeof(fp16));

  float c_conv_result_fp32[BATCH * OUT_H * OUT_W * OUT_C];
  for (int i = 0; i < BATCH * OUT_H * OUT_W * OUT_C; i++) {
    c_conv_result_fp32[i] = __fp16_to_fp32(c_conv_fp16[i]);
  }

  check_conv_result(c_conv_result_fp32, c_conv_real_fp32, BATCH, OUT_H, OUT_W,
                    OUT_C);
}

void test_conv2d_fp16_fp16_fp32() {

  for (int i = 0; i < BATCH * IN_H * IN_W * IN_C; i++) {
    a_conv_fp16[i] = __fp32_to_fp16(a_conv_fp32[i]);
  }
  for (int i = 0; i < OUT_C * FILTER_H * FILTER_W * IN_C; i++) {
    b_conv_fp16[i] = __fp32_to_fp16(b_conv_fp32[i]);
  }

  float c_conv_fp32[BATCH * OUT_H * OUT_W * OUT_C] = {0};

  const long dims_conv[3][4] = {
      {BATCH, IN_H, IN_W, IN_C}, // input: [batch, height, width, channels]
      {OUT_C, FILTER_H, FILTER_W,
       IN_C}, // filter: [out_channels, height, width, in_channels]
      {BATCH, OUT_H, OUT_W,
       OUT_C} // output: [batch, height, width, out_channels]
  };

  const long strides_conv[3][4] = {
      {IN_H * IN_W * IN_C, IN_W * IN_C, IN_C, 1},             // input strides
      {FILTER_H * FILTER_W * IN_C, FILTER_W * IN_C, IN_C, 1}, // filter strides
      {OUT_H * OUT_W * OUT_C, OUT_W * OUT_C, OUT_C, 1}        // output strides
  };

  run_kernel("tx_conv_float_fp16_fp16_fp32", a_conv_fp16, b_conv_fp16,
             c_conv_fp32, dims_conv, strides_conv, sizeof(fp16), sizeof(float));

  check_conv_result(c_conv_fp32, c_conv_real_fp32, BATCH, OUT_H, OUT_W, OUT_C);
}

int main(int argc, char **argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  test_conv2d_fp32_fp32_fp32();
  test_conv2d_fp16_fp16_fp16();
  test_conv2d_fp16_fp16_fp32();

  npu_runtime_deinit(&runtime);
  return 0;
}
