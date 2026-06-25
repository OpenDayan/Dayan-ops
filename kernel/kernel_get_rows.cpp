#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_cache.h"
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
static inline void configLSUTCSR(int m, int n) {

    // NOTE: 新硬件配置寄存器 Stride 数值均以 Bytes 为单位
    long m_bytes = m * sizeof(elemty);
    if(n && ((m_bytes % 64) != 0)) {
        printf("LSU transfer size must be aligned to 64B, but got m_bytes=%ld\n", m_bytes);
    }

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

__device__ long get_rows_f32_i32_f32(void *a, void *b, void *c, int h1, int n1, int m1, int m0,
                                     int n0) {
    // assume the shape fits
    // a: source data, b: index c: result dst
    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)
    configLSUTCSR<lsuTCSR, float>(m0, 1);

    // b(行索引)由 CPU 直接读取,读取前 invalidate 以避免命中 DMA/host 写入前的过期缓存行
    cache_inval_range(b, (size_t)m1 * sizeof(int));

    // 实际trace中只有b只有一个数字，优化掉外面两层循环, m1 = 1
    for (int p = 0; p < m1; ++p) {
        int *b_int = (int *)b;
        int row_idx = b_int[p];
        if (row_idx < 0 || row_idx > n0) {
            printf("[ERROR] get_rows_f32_i32_f32 invalid row_idx value: %d", row_idx);
            return 0;
        }
        tx_lsu_dma_d2d<lsuTCSR>((long)c + p * m0 * sizeof(float),
                                (long)a + row_idx * m0 * sizeof(float));
        __tx_lsu_sb_bar();
    }

    return 0;
}