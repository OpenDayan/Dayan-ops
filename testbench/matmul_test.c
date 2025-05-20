#include <stdio.h>

extern long gmma_e4m3_f32_32_32_32(void* a, void* b, void* c,
    int m,   int n,   int k,
    int lda, int ldb, int ldc);


// A mem               DIM0 DIM1
char a[512 * 256]; // A[512][256]
// B mem
char b[256 * 512]; // B[256][512]
// C mem
char c[512 * 512]; // C[512][512]


int main() {
    printf("Hello, World!\n");
    gmma_e4m3_f32_32_32_32(
          a,   b,   c,
        512, 512, 256,
        512, 256, 512);
}