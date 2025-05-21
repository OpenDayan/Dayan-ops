#include <stdio.h>

#define check(x, expect)                                                                         \
    if ((x) != (expect))                                                                         \
    {                                                                                            \
        printf("Error: %s:%d: %s = %ld, expected %ld\n", __FILE__, __LINE__, #x, (x), (expect)); \
        return -1;                                                                               \
    }

extern long gmma_e4m3_bf16_32_32_32(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);
extern long gmma_s8_s32_32_32_32(void *a, void *b, void *c,
                                 int m, int n, int k,
                                 int lda, int ldb, int ldc);
extern long gmma_e4m3_bf16_32_32_32_AT(void *a, void *b, void *c,
                                       int m, int n, int k,
                                       int lda, int ldb, int ldc);
extern long gmma_s8_s32_32_32_32_AT(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);

extern void __tx_init();

// A mem               DIM0 DIM1
char a[64 * 128] = {
#include "a_col.gen"
}; // A[ 64][128]
// B mem
char b[128 * 96] = {
#include "b_col.gen"
}; // B[128][ 96]
// C mem
int c[64 * 96]; // C[ 64][ 96]

int c_real[64 * 96] = {
#include "c_col.gen"
}; // C[ 64][ 96]

int main()
{
    __tx_init();
    printf("Hello, World!\n");
    gmma_s8_s32_32_32_32(
        a, b, c,
        64, 96, 128,
        64, 128, 64);
    for (int n = 0; n < 96; n++)
    {
        for (int m = 0; m < 64; m++)
        {
            check(c[m * 96 + n], c_real[m * 96 + n]);
        }
    }
    printf("test_gmma_s8_s32_32_32_32 success\n");
}
