#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_vec_csr.hpp"
#include <cmath>
#include <stdio.h>
#include <string.h>
// NOTE: 老一版硬件单独用一套 TCSR，不需要修改
const long aTCSR = 0;
const long bTCSR = 1;
const long cTCSR = 2;
const long oneTCFG = 3;
// NOTE: 新一版硬件 Vec、LSU、MM 各自用一套配置寄存器
const long lsuTCSR = 0; // 使用 0 号 LSU 配置寄存器

// 老寄存器配置不用变
static inline void configTCSRs(int m, int n) {
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE2>(m * n);
    __tx_tcsr_write<aTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<aTCSR * 8 + TCSR_STRIDE3>(m * n);

    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE2>(m * n);
    __tx_tcsr_write<bTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<bTCSR * 8 + TCSR_STRIDE3>(m * n);

    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM0>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM1>(n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE1>(m);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE2>(m * n);
    __tx_tcsr_write<cTCSR * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<cTCSR * 8 + TCSR_STRIDE3>(m * n);
}

static inline void configoneTCSR(void) {
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM0>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM1>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE1>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE2>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<oneTCFG * 8 + TCSR_STRIDE3>(1);
}


// NOTE: 新版硬件 LSU 传输的最低粒度为一个BLOCK(大小固定为64B)，可以理解为把 Tensor 按照 BLOCK 切分进行传输
// NOTE: 对于新版硬件的配置寄存器，一组 TCSR (32个) 作为一个 GROUP，一个 GROUP 的寄存器用于描述一个 Load/Store
//       运算的输入输出形状，模板参数 GROUPBASE 代表 LSU TCSR 标号，elemty 表示原Tensor元素数据类型
template <long GROUPBASE, typename elemty>
static inline void configLSUTCSR(int m, int n) {

    // NOTE: 新硬件配置寄存器 Stride 数值均以 Bytes 为单位
    long m_bytes = m * sizeof(elemty);
    // NOTE: 最小传输粒度是 64B，所以 m_bytes 必须 >= 64，否则 LSU_DIM0 值会溢出
    m_bytes = m_bytes < 64 ? 64 : m_bytes;
    if(n && ((m_bytes % 64) != 0)) {
        // TODO: 不支持非对齐传输
        printf("LSU transfer size must be aligned to 64B, but got m_bytes=%ld\n", m_bytes);
    }

    // 输入/输出 维度，LSU 的输入和输出维度一定是相同的
    // LSU的第0维：LSU_BLOCK 寄存器值表示这一维 BLOCK 的数量，这里设置成0，相当于 原Tensor第0维 合并到 LSU的第1维
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
    // LSU的第1维：相当于 原Tensor第0维(m) 合并到了这一维，这里这里维度大小应该除以块大小，注意 dim大小=寄存器值+1
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(((m_bytes + 63) / 64) - 1);
    // LSU的第2维：相当于 原Tensor第1维(n)
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(n - 1);
    // LSU的第3维：相当于 原Tensor第2维(n)
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);

    // 输入步长
    // LSU第1维步长：传输粒度是 BLOCK，所以步长是每个“元素”的大小，即为 BLOCK=64B
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    // LSU第2维步长：相当于 原Tensor第1维(n) 的步长
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(m_bytes);

    // 输出步长
    // LSU第1维步长：传输粒度是 BLOCK，所以步长是每个“元素”的大小，即为 BLOCK=64B
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    // LSU第2维步长：相当于 原Tensor第1维(n) 的步长
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(m_bytes);

    // 与本算子无关的配置寄存器，但是必须初始化
    // NOTE: BLOCK 不做转置的时候需要将 BLOCKTRANSPOSE初始化为 6，原因详见模拟器源码 rvtng-cmodel/emu/lsu.cpp
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    // BLOCK级别的padding填充：LSU的第0维在 padding 之后的 BLOCK 数量，LSU_BLOCKPADDING - LSU_BLOCK 即为要填充 padding value 的 BLOCK 数量
    // 这里 LSU_BLOCKPADDING 和 LSU_BLOCK 相同 (0个 BLOCK 需要Padding填充)
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    // BLOCK内Byte级别的padding填充：每个BLOCK内填充 padding value 开始的位置(Bytes)。
    // 当前每个 BLOCK 内部都不需要 padding，所以 padding 的开始位置是 BLOCK 末尾，即 63+1=64B 处
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
    // 初始化 Swizzle 相关配置寄存器
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    // LSU的第0维元素的步长，每个元素是一个BLOCK，即为BLOCK间的步长，因为LSU第0维被合并到第1维度(lSU_BLOCK=0)，这里用不到这个寄存器
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
}

// NOTE: 新版硬件可以直接操控 URAM 空间（目前 URAM 总共 1MB）
extern int _uram[256 * 1024];

__device__ long silu_f32(float *x, float *y, int b, int h, int n, int m) {
    // NOTE: 新版硬件暂不支持 get_mcycle 函数
    // uint64_t cycle_begin = get_mcycle();
    const int aTile = 1024;
    const int64_t num = b * h * n * m;

    int op_count = (num - 1) / aTile + 1;

    // NOTE: 新版硬件可以直接在 kernel 内访问 URAM的数据，URAM 基地址由 _uram 指定
    float *XTile[2] = {(float *)_uram, (float *)_uram + aTile};
    float *YTile[2] = {XTile[1] + aTile, XTile[1] + 2 * aTile};
    float *mid1 = YTile[1] + aTile;
    float *mid2 = mid1 + aTile;
    float *minusone = mid2 + aTile;

    // NOTE: 新版硬件可以直接操控 URAM 空间，标量可以直接写入
    // * (float *)(_uram + (((size_t)minusone-(size_t)_uram) / sizeof(float))) = -1.0f;
    *minusone = -1.0f;

    configoneTCSR();

    if (op_count == 1) { // 数据足以单次处理
        configTCSRs(num, 1);
        // NOTE: 在配置老版本寄存器之后一般需要额外配置新版本 LSU 专用配置寄存器，供 load/store 使用
        configLSUTCSR<lsuTCSR, float>(num, 1); // 设置 LSU 专用配置寄存器，tensor形状为 1 x num

        // NOTE: 不再使用 tx_async_uram_load 指令，改用新版异步拷贝指令 tx_lsu_dma_d2u
        // 调用新版异步拷贝指令把数据从DRAM搬运到URAM
        // 注意： DMA 指令的第一个参数为 rd，指定目标地址，第二个参数为 rs1，指定源地址
        tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[0]), (long)x);
        __tx_lsu_sb_bar(); // 新版异步拷贝同步指令


        // NOTE: Vec 类指令依然使用老版本指令
        // 注意：老版本指令的目标地址依然为最后一个参数
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[0], minusone, mid1);
        __tx_sfu_exp_fp32<aTCSR, aTCSR>(mid1, mid2);
        __tx_fp_sub_fp32<aTCSR, oneTCFG, aTCSR, 0b0001>(mid2, minusone, mid1);
        __tx_sfu_recip_fp32<aTCSR, aTCSR>(mid1, mid2);
        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(XTile[0], mid2, YTile[0]);
        __tx_sb_bar(); // 新版同步指令

        // NOTE: 不再使用 tx_async_uram_store 指令，改用新版异步拷贝指令 tx_lsu_dma_u2d
        // 调用新版异步拷贝指令把数据从URAM搬运到DRAM
        // 注意： DMA 指令的第一个参数为 rd，指定目标地址，第二个参数为 rs1，指定源地址
        tx_lsu_dma_u2d<lsuTCSR>((long)y, (long)(YTile[0]));

        // NOTE: 新版硬件不再使用 tx_fence 指令进行同步;
        __tx_lsu_sb_bar(); // 新版异步拷贝同步指令

    } else {
        // load_0
        configTCSRs(aTile, 1);
        configLSUTCSR<lsuTCSR, float>(aTile, 1); // 设置 LSU 专用配置寄存器，tensor形状为 1 x aTile
        tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[0]), (long)x); // 调用新版异步拷贝指令把数据从DRAM搬运到URAM
        __tx_lsu_sb_bar(); // 新版异步拷贝同步指令
        __tx_sb_bar(); // 新版同步指令

        // compute_0
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[0], minusone, mid1);
        __tx_sfu_exp_fp32<aTCSR, aTCSR>(mid1, mid2);
        __tx_fp_sub_fp32<aTCSR, oneTCFG, aTCSR, 0b0001>(mid2, minusone, mid1);
        __tx_sfu_recip_fp32<aTCSR, aTCSR>(mid1, mid2);
        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(XTile[0], mid2, YTile[0]);

        // load_1
        int size = ((num - aTile) > aTile) ? aTile : (num - aTile);
        configTCSRs(size, 1);
        configLSUTCSR<lsuTCSR, float>(size, 1); // 设置 LSU 专用配置寄存器，tensor形状为 1 x size
        tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[1]), (long)(&x[aTile])); // 调用新版异步拷贝指令把数据从DRAM搬运到URAM
        __tx_lsu_sb_bar(); // 新版异步拷贝同步指令
        __tx_sb_bar(); // 新版同步指令

        for (int i = 2; i < op_count; i++) {
            int store_num = i - 2;
            int compute_num = i - 1;
            int load_num = i;

            int store_buffer_num = store_num % 2;
            int compute_buffer_num = compute_num % 2;
            int load_buffer_num = load_num % 2;

            configTCSRs(aTile, 1);
            configLSUTCSR<lsuTCSR, float>(aTile, 1); // 设置 LSU 专用配置寄存器，tensor形状为 1 x aTile

             // store_i-2
            float *y_store = y + store_num * aTile;
            // __tx_async_uram_store<cTCSR, cTCSR, 2>(YTile[store_buffer_num], y_store);
            tx_lsu_dma_u2d<lsuTCSR>((long)y_store, (long)(YTile[store_buffer_num])); // 调用新版异步拷贝指令把数据从URAM搬运到DRAM

            // compute_i-1
            __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[compute_buffer_num], minusone, mid1);
            __tx_sfu_exp_fp32<aTCSR, aTCSR>(mid1, mid2);
            __tx_fp_sub_fp32<aTCSR, oneTCFG, aTCSR, 0b0001>(mid2, minusone, mid1);
            __tx_sfu_recip_fp32<aTCSR, aTCSR>(mid1, mid2);
            __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(XTile[compute_buffer_num], mid2,
                                                          YTile[compute_buffer_num]);

            // load_i
            float *x_load = x + load_num * aTile;
            tx_lsu_dma_d2u<lsuTCSR>((long)(XTile[load_buffer_num]), (long)x_load); // 调用新版异步拷贝指令把数据从DRAM搬运到URAM
            __tx_lsu_sb_bar(); // 新版异步拷贝同步指令
            __tx_sb_bar(); // 新版同步指令
        }

        // store_n-1
        float *y_store_0 = y + (op_count - 2) * aTile;
        tx_lsu_dma_u2d<lsuTCSR>((long)y_store_0, (long)(YTile[op_count % 2])); // 调用新版异步拷贝指令把数据从URAM搬运到DRAM

        // compute_n
        size = num - ((op_count - 1) * aTile);
        configTCSRs(size, 1);
        configLSUTCSR<lsuTCSR, float>(size, 1); // 设置 LSU 专用配置寄存器，tensor形状为 1 x size
        int compute_buffer_num = (op_count - 1) % 2;
        __tx_fp_mul_fp32<aTCSR, oneTCFG, cTCSR, 0b0001>(XTile[compute_buffer_num], minusone, mid1);
        __tx_sfu_exp_fp32<aTCSR, aTCSR>(mid1, mid2);
        __tx_fp_sub_fp32<aTCSR, oneTCFG, aTCSR, 0b0001>(mid2, minusone, mid1);
        __tx_sfu_recip_fp32<aTCSR, aTCSR>(mid1, mid2);
        __tx_fp_mul_fp32<aTCSR, bTCSR, cTCSR, 0b0000>(XTile[compute_buffer_num], mid2,
                                                      YTile[compute_buffer_num]);
        __tx_sb_bar(); // 新版同步指令
        // store_n
        float *y_store_1 = y + (op_count - 1) * aTile;
        tx_lsu_dma_u2d<lsuTCSR>((long)y_store_1, (long)(YTile[(op_count - 1) % 2])); // 调用新版异步拷贝指令把数据从URAM搬运到DRAM
        __tx_lsu_sb_bar(); // 新版异步拷贝同步指令
    }

    return 0;
}

