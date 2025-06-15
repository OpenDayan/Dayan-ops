#include <stdio.h>
#include <stdint.h>

int isnan(float in) {
    int v = *(int*)&in;
    return (v & 0x7f000000) == 0x7f000000;
}

#define check(x, expect)                                                                                     \
    if ((x) != (expect))                                                                                     \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, m, n, #x, (x), (expect)); \
        return -1;                                                                                           \
    }

#define fp_check(x, expect, eps)                                                               \
    if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps))                                      \
    {                                                                                          \
        printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x, (x), (expect)); \
        exit(-1);                                                                             \
    }

#define fp_checkr(m, n, x, expect, eps)                                                                      \
    if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps))                                                    \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %x, expected %x\n", __FILE__, __LINE__, m, n, #x, *(int*)&(x), *(int*)&(expect)); \
    }

extern long gmma_e4m3q_fp32_BT_QK32(void *a, void *b, void *c,
                                    int m, int n, int k,
                                    int lda, int ldb, int ldc);
extern long gmma_e4m3q_fp32_BT_QK256(void *a, void *b, void *c,
                                     int m, int n, int k,
                                     int lda, int ldb, int ldc);
                                    

extern void __tx_init();

#include "generated/e4m3q_mat_size.gen"

// A^T mem               DIM0 DIM1
float a_in[M * K] = {
#include "generated/e4m3q_a_row.gen"
}; // A[M][K]
// B mem
float b_in[N * K] = {
#include "generated/e4m3q_b_col.gen"
}; // B[N][K]
// C mem

// A^T mem               DIM0 DIM1
float as_in[M * K / QK] = {
#include "generated/e4m3q_as_row.gen"
}; // A[M][K]
// B mem
float bs_in[N * K / QK] = {
#include "generated/e4m3q_bs_col.gen"
}; // B[N][K]
// C mem

struct blk_e4m3q
{
    float scale;
    uint8_t v[QK];
};

struct blk_e4m3q a[M * K / QK];
struct blk_e4m3q b[N * K / QK];

float c[N * M];

float c_real[N * M] = {
#include "generated/e4m3q_c_col.gen"
}; // C[N][M]

extern uint8_t __f32_to_e4m3(float f);
extern float __e4m3_to_f32(uint8_t f);

int main()
{
    __tx_init();
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < (K / QK); k++)
        {
            a[m * K / QK + k].scale = as_in[m * K / QK + k];
        }
    }
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < (K / QK); k++)
        {
            b[n * K / QK + k].scale = bs_in[n * K / QK + k];
        }
    }
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            b[(n * K + k) / QK].v[k % QK] = __f32_to_e4m3(b_in[n * K + k]);
            fp_checkr(n, k, __e4m3_to_f32(b[(n * K + k) / QK].v[k % QK]), b_in[n * K + k], 0.0f);
        }
    }
    for (int m = 0; m < M; m++)
    {
        for (int k = 0; k < K; k++)
        {
            a[(m * K + k) / QK].v[k % QK] = __f32_to_e4m3(a_in[m * K + k]);
            fp_checkr(m, k, __e4m3_to_f32(a[(m * K + k) / QK].v[k % QK]), a_in[m * K + k], 0.0f);
        }
    }
    fp_check(a[1].scale, ((float *)a)[1 * (QK / sizeof(float) + 1)], 0.0f);
    printf("gmma_e4m3q_fp32_BT_QK256 start\n");
    gmma_e4m3q_fp32_BT_QK256(
        b, a, c,
        N, M, K,
        K + K * 4 / QK, K + K * 4 / QK, M);
    printf("gmma_e4m3q_fp32_BT_QK256 finish\n");
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            fp_checkr(m, n, c[n * M + m], c_real[n * M + m], 0.0f);
        }
    }
    printf("gmma_e4m3q_fp32_BT_QK256 success\n");
}
