#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static constexpr long setRowsLsuTCSR = 0;

static inline uint16_t fp32_to_bf16_round(float v) {
    union {
        float f;
        uint32_t u;
    } in;
    in.f = v;
    if ((in.u & 0x7fffffffU) > 0x7f800000U) {
        return (uint16_t)((in.u >> 16) | 0x0040U);
    }
    return (uint16_t)((in.u + (0x7fffU + ((in.u >> 16) & 1U))) >> 16);
}

template <long GROUPBASE, typename elemty>
static inline void configLSUTCSR_set_rows(int m, int n) {
    long m_bytes = m * (long)sizeof(elemty);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(((m_bytes + 63) / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(n - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(m_bytes);

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

template <typename elem_t, typename idx_t>
static inline long set_rows_impl(void *src0, void *src1, void *dst, int nc, int nr, int ne1,
                                 int ne02, int ne03, int ne11, int ne12, int nb01, int nb02,
                                 int nb03, int nb10, int nb11, int nb12, int nb1, int nb2,
                                 int nb3) {
    if (nc <= 0) {
        return -1;
    }

    configLSUTCSR_set_rows<setRowsLsuTCSR, elem_t>(nc, 1);

    for (int i03 = 0; i03 < ne03; ++i03) {
        for (int i02 = 0; i02 < ne02; ++i02) {
            const int i12 = i03 % ne12;
            const int i11 = i02 % ne11;

            for (int i = 0; i < nr; ++i) {
                const idx_t i1 = *(idx_t *)((char *)src1 + i * nb10 + i11 * nb11 + i12 * nb12);
                if (i1 < 0 || i1 >= ne1) {
                    printf("[ERROR] invalid row_idx value: %ld\n", (long)i1);
                    return -2;
                }

                const long src_row = (long)src0 + i * nb01 + i02 * nb02 + i03 * nb03;
                const long dst_row = (long)dst + (long)i1 * nb1 + i02 * nb2 + i03 * nb3;

                tx_lsu_dma_d2d<setRowsLsuTCSR>(dst_row, src_row);
                __tx_lsu_sb_bar();
            }
        }
    }

    return 0;
}

template <typename idx_t>
static inline long set_rows_f32_to_bf16_impl(void *src0, void *src1, void *dst, int nc, int nr,
                                             int ne1, int ne02, int ne03, int ne11, int ne12,
                                             int nb01, int nb02, int nb03, int nb10, int nb11,
                                             int nb12, int nb1, int nb2, int nb3) {
    if (nc <= 0) {
        return -1;
    }

    for (int i03 = 0; i03 < ne03; ++i03) {
        for (int i02 = 0; i02 < ne02; ++i02) {
            const int i12 = i03 % ne12;
            const int i11 = i02 % ne11;

            for (int i = 0; i < nr; ++i) {
                const idx_t i1 = *(idx_t *)((char *)src1 + i * nb10 + i11 * nb11 + i12 * nb12);
                if (i1 < 0 || i1 >= ne1) {
                    printf("[ERROR] invalid row_idx value: %ld\n", (long)i1);
                    return -2;
                }

                const float *src_row =
                    (const float *)((char *)src0 + i * nb01 + i02 * nb02 + i03 * nb03);
                uint16_t *dst_row =
                    (uint16_t *)((char *)dst + (long)i1 * nb1 + i02 * nb2 + i03 * nb3);

                for (int j = 0; j < nc; ++j) {
                    dst_row[j] = fp32_to_bf16_round(src_row[j]);
                }
            }
        }
    }

    return 0;
}

__device__ long set_rows_f32_i64_bf16(void *src0, void *src1, void *dst, int nc, int nr, int ne1,
                                      int ne02, int ne03, int ne11, int ne12, int nb01, int nb02,
                                      int nb03, int nb10, int nb11, int nb12, int nb1, int nb2,
                                      int nb3) {
    return set_rows_f32_to_bf16_impl<int64_t>(src0, src1, dst, nc, nr, ne1, ne02, ne03, ne11, ne12,
                                              nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3);
}