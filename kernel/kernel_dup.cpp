#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long aUramTCSR = 3;
const long cUramTCSR = 4;
const long lsuTCSR0 = 0;
const long lsuTCSR1 = 1;

extern int _uram[256 * 1024];

#define aTile 1024

static inline void configTCSR_Tile_cont_2d(int m, int n) {
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_STRIDE2>(m*n);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aUramTCSR * 8 + TCSR_STRIDE3>(m*n);

    __tx_tcsr_write<cUramTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_STRIDE2>(m*n);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cUramTCSR * 8 + TCSR_STRIDE3>(m*n);
}

static inline void configTCSR_Tile_cont(int elem_per_tile, int last_elem_tile_num) {
    // a存每组长度，b存最后一组长度
    // printf("configTCSR_Tile_cont elem_per_tile = %d, last_elem_tile_num = %d\n", elem_per_tile,
    // last_elem_tile_num);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(elem_per_tile);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(elem_per_tile);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(elem_per_tile);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(elem_per_tile);

    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(last_elem_tile_num);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(last_elem_tile_num);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(last_elem_tile_num);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(last_elem_tile_num);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(elem_per_tile);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(elem_per_tile);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(elem_per_tile);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(elem_per_tile);
}

template <long GROUPBASE, typename elemty>
static inline void configLSUTCSRd2u(int m, int n, int n_stride_bytes) {
    long m_bytes = m * sizeof(elemty);
    if(n && ((m_bytes % 64) != 0)) {
        // TODO: 不支持非对齐传输
        printf("D2U transfer size must be aligned to 64B, but got n_stride_bytes=%ld\n", n_stride_bytes);
    }

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(((m_bytes + 63) / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(n - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(n_stride_bytes);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(m_bytes);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
}

template <long GROUPBASE, typename elemty>
static inline void configLSUTCSRu2d(int m, int n, int n_stride_bytes) {
    long m_bytes = m * sizeof(elemty);
    m_bytes = m_bytes < 64 ? 64 : m_bytes;
    if(n && ((m_bytes % 64) != 0)) {
        // TODO: 不支持非对齐传输
        printf("U2D transfer size must be aligned to 64B, but got n_stride_bytes=%ld\n", n_stride_bytes);
    }

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(((m_bytes + 63) / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(n - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(m_bytes);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(n_stride_bytes);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
}

static inline int check_dim(int a_bs, int a_h, int a_n, int a_m, int res_bs, int res_h, int res_n,
                            int res_m) {
    if (a_bs != res_bs) {
        printf("Dim-3 is different between src(%d) and dst(%d)\n", a_bs, res_bs);
        return -1;
    }
    if (a_h != res_h) {
        printf("Dim-2 is different between src(%d) and dst(%d)\n", a_h, res_h);
        return -1;
    }
    if (a_n != res_n) {
        printf("Dim-1 is different between src(%d) and dst(%d)\n", a_n, res_n);
        return -1;
    }
    if (a_m != res_m) {
        printf("Dim-0 is different between src(%d) and dst(%d)\n", a_m, res_m);
        return -1;
    }
    return 0;
}

__device__ long dup_cont_f32_cont_f32(void *a, void *res, int a_bs, int a_h, int a_n, int a_m,
                                      int res_bs, int res_h, int res_n, int res_m) {
    // 要求 (a_bs*a_h*a_n*a_m) == (res_bs*res_h*res_n*res_m)
    // 分块处理
    // 假设连续时(bs, h, n, m)和(1, 1, 1, bs*h*n*m)等价
    int total_elem_num = a_bs * a_h * a_n * a_m;
    int elem_per_tile = (total_elem_num > aTile) ? aTile : total_elem_num;
    int last_elem_tile_num =
        (total_elem_num % elem_per_tile == 0) ? elem_per_tile : (total_elem_num % elem_per_tile);
    configTCSR_Tile_cont(elem_per_tile, last_elem_tile_num);
    configLSUTCSRd2u<lsuTCSR0, float>(elem_per_tile, 1, elem_per_tile*sizeof(float));

    int num_elem_copied;
    for (num_elem_copied = 0; num_elem_copied + elem_per_tile <= total_elem_num;
         num_elem_copied += elem_per_tile) {
        void *a_ptr = (char *)a + num_elem_copied * sizeof(float);
        void *res_ptr = (char *)res + num_elem_copied * sizeof(float);
        tx_lsu_dma_d2d<lsuTCSR0>((long)res_ptr, (long)a_ptr);
        __tx_lsu_sb_bar();
    }
    if (total_elem_num - num_elem_copied > 0) {
        configLSUTCSRd2u<lsuTCSR0, float>(last_elem_tile_num, 1, last_elem_tile_num*sizeof(float));
        void *a_ptr = (char *)a + num_elem_copied * sizeof(float);
        void *res_ptr = (char *)res + num_elem_copied * sizeof(float);
        tx_lsu_dma_d2d<lsuTCSR0>((long)res_ptr, (long)a_ptr);
        __tx_lsu_sb_bar();
    }

    return 0;
}

__device__ long dup_cont_f32_cont_bf16(void *a, void *res, int a_bs, int a_h, int a_n, int a_m,
                                       int res_bs, int res_h, int res_n, int res_m) {
    uint64_t uram0 = (uint64_t)((char *)_uram + 0x10000);
    uint64_t uram1 = (uint64_t)((char *)_uram + 0x20000);
    if (a_m * sizeof(float) > 0x10000) {  // 按行拷贝，每行大小不得大于 64KB，即 a_m < 16K
        printf("a_m is too big!\n");
        return -1;
    }
    // 要求 (a_bs*a_h*a_n*a_m) == (res_bs*res_h*res_n*res_m)
    // 分块处理
    // 假设连续时(bs, h, n, m)和(1, 1, 1, bs*h*n*m)等价
    int total_elem_num = a_bs * a_h * a_n * a_m;
    int elem_per_tile = a_m;
    int last_elem_tile_num =
        (total_elem_num % elem_per_tile == 0) ? elem_per_tile : (total_elem_num % elem_per_tile);
    configTCSR_Tile_cont(elem_per_tile, last_elem_tile_num);
    configLSUTCSRd2u<lsuTCSR0, float>(elem_per_tile, 1, elem_per_tile*sizeof(float));
    configLSUTCSRu2d<lsuTCSR1, uint16_t>(elem_per_tile, 1, elem_per_tile*sizeof(uint16_t));

    int num_elem_copied;
    for (num_elem_copied = 0; num_elem_copied + elem_per_tile <= total_elem_num;
         num_elem_copied += elem_per_tile) {
        void *a_ptr = (char *)a + num_elem_copied * sizeof(float);
        void *res_ptr = (char *)res + num_elem_copied * sizeof(uint16_t);
        tx_lsu_dma_d2u<lsuTCSR0>((long)uram0, (long)a_ptr);
        __tx_lsu_sb_bar();
        __tx_convert_fp32_bf16_round<aTCSR, cTCSR>((void *)uram0, (void *)uram1);
        __tx_sb_bar();
        tx_lsu_dma_u2d<lsuTCSR1>((long)res_ptr, (long)uram1);
        __tx_lsu_sb_bar();
    }

    return 0;
}

static inline long dup_f32_bf16_rows(void *a, void *res, int a_bs, int a_h, int a_n, int a_m,
                                int res_bs, int res_h, int res_n, int res_m, int a_nb3,
                                int a_nb2, int a_nb1, int a_nb0, int res_nb3, int res_nb2,
                                int res_nb1, int res_nb0) {
    uint64_t uram0 = (uint64_t)((char *)_uram + 0x10000);
    uint64_t uram1 = (uint64_t)((char *)_uram + 0x20000);
    // 按行拷贝时，要求输入输出每个维度大小相同
    if ((a_bs != res_bs) || (a_h != res_h) || (a_n != res_n) || (a_m != res_m)) {
        return -1;
    }

    // each row of res
    configTCSR_Tile_cont_2d(a_m, a_n);

    // configLSUTCSRd2u<lsuTCSR0, float>(a_m, a_n, a_nb1);
    // configLSUTCSRu2d<lsuTCSR1, uint16_t>(res_m, res_n, res_nb1);
    printf("DIM: %d %d %d %d\n", a_bs, a_h, a_n, a_m);
    printf("A Stride: %d %d %d %d\n", a_nb3, a_nb2, a_nb1, a_nb0);
    printf("R Stride: %d %d %d %d\n", res_nb3, res_nb2, res_nb1, res_nb0);

    for (int bs = 0; bs < a_bs; ++bs) {
        for (int i = 0; i < a_h; ++i) {
            // tx_lsu_dma_d2u<lsuTCSR0>((long)uram0, (long)(a + i * a_nb2 + bs * a_nb3));
            // __tx_lsu_sb_bar();
            for (int j = 0 ; j < a_n; ++j) {
                for(int k = 0 ; k < a_m; ++ k) {
                    uint32_t raw_float = *(uint32_t*)((long)a + k * a_nb0 + j * a_nb1 + i * a_nb2 + bs * a_nb3);
                    ((uint32_t*)uram0)[k + j * a_m] = raw_float;
                }
            }
            __tx_convert_fp32_bf16_round<aUramTCSR, cUramTCSR>((void *)uram0, (void *)uram1);
            // __tx_sb_bar();
            // tx_lsu_dma_u2d<lsuTCSR1>((long)(res + i * res_nb2 + bs * res_nb3), (long)uram1);
            // __tx_lsu_sb_bar();
            for (int j = 0 ; j < a_n; ++j) {
                for(int k = 0 ; k < a_m; ++ k) {
                    *(uint16_t*)((long)res + k * res_nb0 + j * res_nb1 + i * res_nb2 + bs * res_nb3) =
                    ((uint16_t*)uram1)[k + j * a_m];
                }
            }
        }
    }
    __asm__ __volatile__(".insn 0xfc000073" ::: "memory"); // CFLUSH.D.L1
    return 0;
}

__device__ long dup_cont_f32_no_cont_bf16(void *a, void *res, int a_bs, int a_h, int a_n, int a_m,
                                          int res_bs, int res_h, int res_n, int res_m, int a_nb3,
                                          int a_nb2, int a_nb1, int a_nb0, int res_nb3, int res_nb2,
                                          int res_nb1, int res_nb0) {
    return dup_f32_bf16_rows(a, res, a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m,
                             a_nb3, a_nb2, a_nb1, a_nb0, res_nb3, res_nb2, res_nb1, res_nb0);
}

__device__ long dup_no_cont_f32_no_cont_bf16(void *a, void *res, int a_bs, int a_h, int a_n,
                                             int a_m, int res_bs, int res_h, int res_n, int res_m,
                                             int a_nb3, int a_nb2, int a_nb1, int a_nb0,
                                             int res_nb3, int res_nb2, int res_nb1, int res_nb0) {
    return dup_f32_bf16_rows(a, res, a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m,
                             a_nb3, a_nb2, a_nb1, a_nb0, res_nb3, res_nb2, res_nb1, res_nb0);
}