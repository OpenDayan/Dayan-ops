#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <stdio.h>
#include <string.h>
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long lsuTCSR = 0;

extern int _uram[256 * 1024];

template <long GROUPBASE, typename elemty>
static inline void configLSUTCSR(int bs, int h, int n, int m) {
    long m_bytes = m * sizeof(elemty);
    if(n && ((m_bytes % 64) != 0)) {
        printf("LSU transfer size must be aligned to 64B, but got m_bytes=%ld\n", m_bytes);
    }

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(((m_bytes + 63) / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(n - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(h - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(bs - 1);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(m_bytes);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(m_bytes * n);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM2STRIDE>(m_bytes * n * h);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(m_bytes * h);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(m_bytes);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(m_bytes * n * h);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(((m_bytes + 63) / 64) - 1);

    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
}

__device__ long permute_f32_2_1(void *a, void *res, int bs, int h, int n, int m) {
    configLSUTCSR<lsuTCSR, float>(bs, h, n, m);
    tx_lsu_dma_d2d<lsuTCSR>((long)res, (long)a);
    __tx_lsu_sb_bar();
    return 0;
}
