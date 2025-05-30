#include <stdio.h>
#include <stdint.h>

#define check(x, expect)                                                                                     \
    if ((x) != (expect))                                                                                     \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, m, n, #x, (x), (expect)); \
        return -1;                                                                                           \
    }

#define fp_check(x, expect, eps)                                                                 \
    if ((x) < (expect) - (eps) || (x) > (expect) + (eps))                                                     \
    {                                                                                                        \
        printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x, (x), (expect)); \
        return -1;                                                                                           \
    }

extern long gmma_e4m3_f32_16_16_16(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);
extern long gmma_s8_s32_16_16_16(void *a, void *b, void *c,
                                 int m, int n, int k,
                                 int lda, int ldb, int ldc);
extern long gmma_e4m3_f32_16_16_16_AT(void *a, void *b, void *c,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc);
extern long gmma_s8_s32_16_16_16_AT(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);

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
    for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
            a[m * K + k] = __f32_to_e4m3(a_in[m * K + k]);
            fp_check(__e4m3_to_f32(a[m * K + k]), a_in[m * K + k], 0.0f);
        }
    }
    for(int n = 0; n < N; n++) {
        for(int k = 0; k < K; k++) {
            b[n * K + k] = __f32_to_e4m3(b_in[n * K + k]);
            fp_check(__e4m3_to_f32(b[n * K + k]), b_in[n * K + k], 0.0f);
        }
    }
    printf("gmma_e4m3_f32_16_16_16_AT start\n");
    gmma_e4m3_f32_16_16_16_AT(
        a, b, c,
        M, N, K,
        K, K, M);
        printf("gmma_e4m3_f32_16_16_16_AT finish\n");
    for(int n = 0; n < N; n++) {
        for(int m = 0; m < M; m++) {
            fp_check(c[n * M + m], c_real[n * M + m], 0.0f);
        }
    }
    printf("gmma_e4m3_f32_16_16_16_AT success\n");
}
