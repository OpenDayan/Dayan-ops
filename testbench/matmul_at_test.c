#include <stdio.h>

#define check(x, expect)                                                                                     \
    if ((x) != (expect))                                                                                     \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, m, n, #x, (x), (expect)); \
        return -1;                                                                                           \
    }

extern long gmma_e4m3_bf16_16_16_16(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);
extern long gmma_s8_s32_16_16_16(void *a, void *b, void *c,
                                 int m, int n, int k,
                                 int lda, int ldb, int ldc);
extern long gmma_e4m3_bf16_16_16_16_AT(void *a, void *b, void *c,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc);
extern long gmma_s8_s32_16_16_16_AT(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);

extern void __tx_init();

#include "int_mat_size.gen"

// A^T mem               DIM0 DIM1
char a[M * K] = {
#include "int_a_row.gen"
}; // A[M][K]
// B mem
char b[N * K] = {
#include "int_b_col.gen"
}; // B[N][K]
// C mem
int c[N * M];

int c_real[N * M] = {
#include "int_c_col.gen"
}; // C[N][M]

int main()
{
    __tx_init();
    printf("test_gmma_s8_s32_16_16_16_AT start\n");
    gmma_s8_s32_16_16_16_AT(
        a, b, c,
        M, N, K,
        K, K, M);
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            check(c[n * M + m], c_real[n * M + m]);
        }
    }
    printf("test_gmma_s8_s32_16_16_16_AT success\n");
}
