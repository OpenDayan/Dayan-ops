#pragma once

enum LSU_TCSR {
    LSU_BLOCK = 0,
    LSU_DIM0  = 1,
    LSU_DIM1  = 2,
    LSU_DIM2  = 3,
    LSU_INBLOCKSTRIDE = 4,
    LSU_INDIM0STRIDE = 5,
    LSU_INDIM1STRIDE = 6,
    LSU_INDIM2STRIDE = 7,
    LSU_OUTBLOCKSTRIDE = 8,
    LSU_OUTDIM0STRIDE = 9,
    LSU_OUTDIM1STRIDE = 10,
    LSU_OUTDIM2STRIDE = 11,
    LSU_BLOCKTRANSPOSE = 12,
    LSU_BLOCKPADDING = 13,
    LSU_REQUESTPADDING = 14,
    LSU_PADDINGVALUE = 15,
    LSU_URAMWRITESWIZZLEMBS = 16,
    LSU_URAMREADSWIZZLEMBS = 17,
    LSU_CHECKSUM = 31
};

#define LSU_GROUPSZ  (32)

// 便捷宏，配置 LSU 矩阵为指定尺寸，注意，minor 应该对齐到 64
template<long GROUPBASE>
void set_lsu_matrix(long &cksum, long dim1, long dim0, long inStride, long outStride) {
    // A Configuration
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>((dim0 / 64) - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(dim1 - 1);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(inStride);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(outStride);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    // Checksum ignored for now.
    cksum = 0;
}
