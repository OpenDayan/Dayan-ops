#include <stdint.h>
#include <stdio.h>

typedef union {
  float f32;
  int32_t i32;
} fp_bits_t;

int isnan(float in) {
  int v = *(int*)&in;
  return (v & 0x7f000000) == 0x7f000000;
}

#define fp_check(x, expect, eps)                                               \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {                      \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    printf("Error: %s:%d: %s = %08x, expected %08x\n", __FILE__, __LINE__, #x, \
           (x_bits.i32), (expect_bits.i32));                                   \
    return -1;                                                                 \
  }

#define fp_check2(x, expect, eps)                                              \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {                      \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    printf("Error: %s:%d: (%d,%d) %s = %08x, expected %08x\n", __FILE__,       \
           __LINE__, m, n, #x, (x_bits.i32), (expect_bits.i32));               \
    return -1;                                                                 \
  }

extern long gmma_e4m3_f32_16_16_16(void *a, void *b, void *c, int m, int n,
                                   int k, int lda, int ldb, int ldc);
extern long gmma_s8_s32_16_16_16(void *a, void *b, void *c, int m, int n, int k,
                                 int lda, int ldb, int ldc);
extern long gmma_e4m3_f32_16_16_16_BT(void *a, void *b, void *c, int m, int n,
                                      int k, int lda, int ldb, int ldc);
extern long gmma_s8_s32_16_16_16_BT(void *a, void *b, void *c, int m, int n,
                                    int k, int lda, int ldb, int ldc);

extern void __tx_init();

#include "generated/e4m3_mat_size.gen"

// A^T mem               DIM0 DIM1
float a_in[M * K] = {
#include "generated/e4m3_a_row.gen"
}; // A[M][K]
// B mem
float b_in[N * K] = {
#include "generated/e4m3_b_col.gen"
}; // B[N][K]
// C mem

uint8_t a[M * K];
uint8_t b[N * K];
float c[N * M];

float c_real[N * M] = {
#include "generated/e4m3_c_col.gen"
}; // C[N][M]

extern uint8_t __f32_to_e4m3(float f);
extern float __e4m3_to_f32(uint8_t f);

int main() {
  __tx_init();
  printf("prepare data\n");
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      a[m * K + k] = __f32_to_e4m3(a_in[m * K + k]);
      fp_check(__e4m3_to_f32(a[m * K + k]), a_in[m * K + k], 0.0f);
    }
  }
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b[n * K + k] = __f32_to_e4m3(b_in[n * K + k]);
      fp_check(__e4m3_to_f32(b[n * K + k]), b_in[n * K + k], 0.0f);
    }
  }
  printf("M: %d, N: %d, K: %d\n", M, N, K);
  printf("gmma_e4m3_f32_16_16_16_BT start\n");
  gmma_e4m3_f32_16_16_16_BT(b, a, c, N, M, K, K, K, M);
  printf("gmma_e4m3_f32_16_16_16_BT finish\n");

  printf("c_col:\n");
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      printf("%x ", ((uint32_t *)c)[i * M + j]);
    }
    printf("\n");
  }

  printf("c_col_golden:\n");
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      printf("%x ", ((uint32_t *)c_real)[i * M + j]);
    }
    printf("\n");
  }

  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      fp_check2(c[n * M + m], c_real[n * M + m], 0.1f);
    }
  }
  printf("gmma_e4m3_f32_16_16_16_BT success\n");
}
