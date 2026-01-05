// BF16 矩阵乘法算子核

/*
* TODO: 优化点
* 1. 优化 Tile Size 等小参
* 2. 目前仅支持 B-K-M-N / B-K-N-M 两种宏遍历顺序，对 K 较大但 MN 较小的情况不能有效复用 C
* 3. 对于多 Batch 的支持，目前较为局限。
*/

#include "npu_instruction.hpp"
#include "npu_instruction_old.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_mat_csr.hpp"
#include "npu_task.hpp"
#include "npu_vec_csr.hpp"
#include <stdio.h>
#include <string.h>

// 控制调试信息打印
#define debugf(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
// #define debugf(fmt, ...) do {} while (0)

// Note0：所有 kernel 内部使用的非递归函数建议修饰为 static inline
// 递归函数建议修饰为 static
// 建议仅标记为 __device__ 的函数对外部可见，以避免命名冲突的可能

// Note1：避免过度抽象封装，将重复的/结构性的代码段或者语句组织抽象为函数
// 如若仅出现一次的代码段或者语句，没有必要单独抽象为一个函数

// Producer-Consumer 编程模型

namespace {
const int LEGACY_A = 0;
const int LSU_A = 0;
const int LSU_B = 1;
const int LSU_CLD = 2;
const int LSU_CST = 3;

// 全局生产者任务板
template <long tileM, long tileN, long tileK, bool ZT> class MatmulPubState {
  public:
    // 任务尺寸
    // 注意：B 表示 Batch
    const int M, N, K, B;
    // 切片个数
    const int cM, cN, cK;
    // 基地址
    const uint64_t aBase, bBase, cBase;
    const uint64_t aStride, bStride, cStride;
    const uint64_t aBatchStride, bBatchStride, cBatchStride;
    NPUURAM *a_buf;
    NPUURAM *b_buf;
    NPUURAM *c_buf;
    const int max_cbuf;
    MatmulPubState(int M, int N, int K, int B, uint64_t aBase, uint64_t bBase, uint64_t cBase,
                   uint64_t aStride, uint64_t bStride, uint64_t cStride,
                   uint64_t aBatchStride, uint64_t bBatchStride, uint64_t cBatchStride,
                   NPUURAM *a_buf, NPUURAM *b_buf, NPUURAM *c_buf, int max_cbuf)
        : M(M), N(N), K(K), B(B),
        aBase(aBase), bBase(bBase), cBase(cBase),
        aStride(aStride), bStride(bStride), cStride(cStride),
        aBatchStride(aBatchStride), bBatchStride(bBatchStride), cBatchStride(cBatchStride),
        a_buf(a_buf), b_buf(b_buf), c_buf(c_buf), max_cbuf(max_cbuf),
        cM((M + tileM - 1) / tileM), cN((N + tileN - 1) / tileN), cK((K + tileK - 1) / tileK){};
    MatmulPubState(const MatmulPubState &other)
        : M(other.M), N(other.N), K(other.K), B(other.B),
        aBase(other.aBase), bBase(other.bBase), cBase(other.cBase),
        aStride(other.aStride), bStride(other.bStride), cStride(other.cStride),
        aBatchStride(other.aBatchStride), bBatchStride(other.bBatchStride), cBatchStride(other.cBatchStride),
        a_buf(other.a_buf), b_buf(other.b_buf), c_buf(other.c_buf), max_cbuf(other.max_cbuf),
        cM(other.cM), cN(other.cN), cK(other.cK){};

  private:
    // 内部状态
    void *advance_state = 0;
    // 循环变量
    int block_row, block_col, m, n, k, b, rst_id, cur_id, cid;

  public:
    // 返回值表示当前任务是否已结束
    inline bool advance(
        int &cur_m, int &cur_n, int &cur_k, int &cur_b,
        int &a_buf_id, int &b_buf_id, int &c_buf_id) {
        if (advance_state)
            goto *advance_state;
        cur_id = 0;
        cid = 0;
        for (b = 0; b < B; b++) {
            for (k = 0; k < cK; k++) {
                if (ZT) {
                    // 列 -> 行 遍历顺序
                    for (block_col = 0; block_col < cN; block_col += 2) {
                        rst_id = cur_id;
                        for (block_row = 0; block_row < cM; block_row += 2) {
                            for (m = block_row; (m < cM) && (m < block_row + 2); m++) {
                                for (n = block_col; (n < cN) && (n < block_col + 2); n++) {
                                    cur_m = m;
                                    cur_n = n;
                                    cur_k = k;
                                    cur_b = b;
                                    a_buf_id = cur_id;
                                    // 【改进点】：结合 n 的坐标和 batch 的偏移
                                    // b_buf_id = n & 0x1;
                                    b_buf_id = (n & 0x1) ^ ((cN & 1) ? (b & 1) : 0);
                                    c_buf_id = cid;
                                    cid = (cid + 0x1) % max_cbuf;
                                    advance_state = &&ZTNextLoop;
                                    return false;
                                ZTNextLoop:;
                                }
                                cur_id ^= 1;
                            }
                            cur_id = rst_id;
                        }
                        cur_id ^= 1;
                    }
                } else {
                    // 行 -> 列 遍历顺序
                    for (block_row = 0; block_row < cM; block_row += 2) {
                        rst_id = cur_id;
                        for (block_col = 0; block_col < cN; block_col += 2) {
                            for (m = block_row; (m < cM) && (m < block_row + 2); m++) {
                                for (n = block_col; (n < cN) && (n < block_col + 2); n++) {
                                    cur_m = m;
                                    cur_n = n;
                                    cur_k = k;
                                    cur_b = b;
                                    // 【改进点】：结合 m 的坐标和 batch 的偏移
                                    // a_buf_id = m & 0x1;
                                    a_buf_id = ((m & 0x1) ^ 1) ^ ((cM & 1) ? (b & 1) : 0);
                                    b_buf_id = cur_id;
                                    c_buf_id = cid;
                                    cur_id ^= 0x1;
                                    cid = (cid + 0x1) % max_cbuf;
                                    advance_state = &&ZNextLoop;
                                    return false;
                                ZNextLoop:;
                                }
                                cur_id = rst_id;
                            }
                        }
                        cur_id ^= 0x1;
                    }
                }
            }
        }
        advance_state = &&Exit;
    Exit:
        return true;
    }
};

// 公共基本方法
template <ulong TCSR_ID>
static inline void load_matrix(
    // Major: 主序维 (K)
    // Minor: 非主序维 (M / N)
    ulong major_left, ulong minor_left, ulong major_tile, ulong minor_tile, ulong uram_base,
    ulong dram_base) {
    if (major_left >= major_tile) {
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>(major_tile / 64 - 1);
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM1>(
            (minor_left < minor_tile ? minor_left : minor_tile) - 1);
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
        tx_lsu_dma_d2u<TCSR_ID>(uram_base, dram_base);
    } else {
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM1>(
            (minor_left < minor_tile ? minor_left : minor_tile) - 1);
        while (major_left) {
            ulong handle_bytes = major_left & (~0x3FUL); // 64 Bytes 对齐
            ulong remain_bytes = major_left & (0x3FUL);
            if (handle_bytes == 0 && remain_bytes > 0) {
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>(0);
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_REQUESTPADDING>(remain_bytes - 1);
                handle_bytes = remain_bytes;
            } else {
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>((handle_bytes / 64) - 1);
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
            }
            tx_lsu_dma_d2u<TCSR_ID>(uram_base, dram_base);
            uram_base += handle_bytes;
            dram_base += handle_bytes;
            major_left = major_left - handle_bytes;
        }
    }
}

template <ulong TCSR_ID, ulong TRANSPOSE_LOG2ELEMSIZE = 1>
static inline void load_matrix_transposed(
    // K: 内积维度
    // MN: 非内积维 (M / N)
    ulong k_left, ulong mn_left, ulong k_tile, ulong mn_tile, ulong uram_base,
    ulong dram_base) {
    const long sub_block_size = 64 / (1 << TRANSPOSE_LOG2ELEMSIZE);
    // 获取真实加载的 mn/k 维度（dim0/dim1）的尺寸
    ulong mn_size = mn_left < mn_tile ? mn_left : mn_tile;
    ulong k_size = k_left < k_tile ? k_left : k_tile;
    if((mn_size % sub_block_size) != 0 || (k_size % sub_block_size) != 0) {
        printf("Not expected alignment!\n");
    }
    // 此时已保证 mn_size == x * sub_block_size, k_size == x * sub_block_size
    tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>(((mn_size * (1 << TRANSPOSE_LOG2ELEMSIZE)) / 64) - 1);
    tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM1>((k_size / sub_block_size) - 1);
    tx_lsu_dma_d2u<TCSR_ID>(uram_base, dram_base);
}

template <ulong TCSR_ID>
static inline void store_matrix(
    // Major: 主序维 (K)
    // Minor: 非主序维 (M / N)
    ulong major_left, ulong minor_left, ulong major_tile, ulong minor_tile, ulong uram_base,
    ulong dram_base) {
    if(major_left % 64 != 0) {
        printf("Warning: store_matrix major_left not aligned to 64 Bytes: %lu\n", major_left);
    }
    if (major_left >= major_tile) {
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>(major_tile / 64 - 1);
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM1>(
            (minor_left < minor_tile ? minor_left : minor_tile) - 1);
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
        tx_lsu_dma_u2d<TCSR_ID>(dram_base, uram_base);
    } else {
        tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM1>(
            (minor_left < minor_tile ? minor_left : minor_tile) - 1);
        while (major_left) {
            ulong handle_bytes = major_left & (~0x3FUL); // 64 Bytes 对齐
            ulong remain_bytes = major_left & (0x3FUL);
            if (handle_bytes == 0 && remain_bytes > 0) {
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>(0);
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_REQUESTPADDING>(remain_bytes - 1);
                handle_bytes = remain_bytes;
            } else {
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_DIM0>((handle_bytes / 64) - 1);
                tx_lsu_tcsr_write<TCSR_ID * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);
            }
            tx_lsu_dma_u2d<TCSR_ID>(dram_base, uram_base);
            uram_base += handle_bytes;
            dram_base += handle_bytes;
            major_left = major_left - handle_bytes;
        }
    }
}

template<long TCSR_ID>
static inline void set_legacy_matrix(int M, int K) {
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM0>(K);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE0>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM1>(M);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE1>(K);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM2>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE2>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_DIM3>(1);
    __tx_tcsr_write<TCSR_ID * 8 + TCSR_STRIDE3>(1);
}

// Producer Task 类
template <long GROUPBASE, long tileM, long tileN, long tileK, bool ZT, long swizzle, bool A_IS_FP32, bool B_TRANSPOSE>
class MatmulProducerTask : public NPUTask {
  private:
    long aCkSum, bCkSum, cCkSum; // tcsr 配置 cksum
    MatmulPubState<tileM, tileN, tileK, ZT> *s;

    inline void config_tcsr() {
        // Note: k and tileK should be aligned to 64-Bytes
        // 暂不考虑 K pad 问题。
        // A Configuration
        set_lsu_matrix<GROUPBASE + LSU_A>(aCkSum, tileM, tileK * 4, s->aStride, tileK * 4);
        // B Configuration
        if(B_TRANSPOSE) set_lsu_matrix<GROUPBASE + LSU_B>(bCkSum, tileN, tileK * 2, s->bStride, tileK * 2);
        else set_lsu_matrix<GROUPBASE + LSU_B, true, 1>(bCkSum, tileN, tileK * 2, s->bStride, tileK * 2);
        // C Configuration
        set_lsu_matrix<GROUPBASE + LSU_CLD>(cCkSum, tileM, tileN * 4, s->cStride, tileN * 4);
        // 设置 A 的加载 Swizzle 为 0，以配合后续转数操作
        tx_lsu_tcsr_write<(GROUPBASE + LSU_A) * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(A_IS_FP32 ? 0 : swizzle);
        tx_lsu_tcsr_write<(GROUPBASE + LSU_B) * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(swizzle);
        tx_lsu_tcsr_write<(GROUPBASE + LSU_CLD) * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(swizzle);
        // set_legacy_matrix<LEGACY_A>(tileM, tileK);
        return;
    }
    // 局部变量
    int m, n, k, batch; // 当前处理中的 m n k batch
    NPUURAM *a, *b, *c;
    int a_buf_id, b_buf_id, c_buf_id;

  public:
    MatmulProducerTask(MatmulPubState<tileM, tileN, tileK, ZT> *s) : s(s), NPUTask(wrapper){};
    int32_t execute(int *err) {
        NPUBegin;
        config_tcsr();
        while (1) {
            // 获取任务
            if (s->advance(m, n, k, batch, a_buf_id, b_buf_id, c_buf_id))
                break;
            debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: PRODUCER GET TASK!\n",
                get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
            // 获取并加载 URAM Buffer
            c = &(s->c_buf[c_buf_id]);
            do {
                NPUWaitFree(c);
            } while (c->allocate(sched, 1, 1));
            if (ZT) {
                if ((n % 2) == 0) {
                    a = &(s->a_buf[a_buf_id]);
                    do {
                        NPUWaitFree(a);
                    } while (a->allocate(sched, 1, (s->cN - n) > 2 ? 2 : (s->cN - n)));
                }
                if (m == 0) {
                    // 负责加载 B Buffer
                    b = &(s->b_buf[b_buf_id]);
                    do {
                        NPUWaitFree(b);
                    } while (b->allocate(sched, 1, s->cM));
                }
            } else {
                if ((m % 2) == 0) {
                    b = &(s->b_buf[b_buf_id]);
                    do {
                        NPUWaitFree(b);
                    } while (b->allocate(sched, 1, (s->cM - m) > 2 ? 2 : (s->cM - m)));
                }
                if (n == 0) {
                    // 负责加载 A Buffer
                    a = &(s->a_buf[a_buf_id]);
                    do {
                        NPUWaitFree(a);
                    } while (a->allocate(sched, 1, s->cN));
                }
            }
            debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: PRODUCER GET FREE URAM BUFFER\n",
                get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
            // 注意，局部变量在下一个 bar 到来时就失效
            if (ZT) {
                if (m == 0) {
                    // 更新 n 和 k
                    // tx_lsu_dma_d2u<GROUPBASE + LSU_B>(b->uram, s->bBase + k * tileK * 2 +
                    //                                            n * tileN * s->bStride);
                    long leftK = s->K - tileK * k;
                    long leftN = s->N - tileN * n;
                    if(B_TRANSPOSE) load_matrix<GROUPBASE + LSU_B>(
                        leftK * sizeof(uint16_t), leftN,
                        tileK * sizeof(uint16_t), tileN,
                        b->uram, s->bBase + k * tileK * sizeof(uint16_t) + n * tileN * s->bStride + batch * s->bBatchStride);
                    else load_matrix_transposed<GROUPBASE + LSU_B, 1 /* LOG2(Sizeof(BFloat16)) == 1 */>(
                        leftK, leftN,
                        tileK, tileN,
                        b->uram, s->bBase + k * tileK * s->bStride + n * tileN * sizeof(uint16_t) + batch * s->bBatchStride);
                }
                if ((n % 2) == 0) {
                    // 更新 m 和 k
                    // tx_lsu_dma_d2u<GROUPBASE + LSU_A>(a->uram, s->aBase + k * tileK * 4 +
                    //                                            m * tileM * s->aStride);
                    if(A_IS_FP32) {
                        long leftK = s->K - tileK * k;
                        long leftM = s->M - tileM * m;
                        load_matrix<GROUPBASE + LSU_A>(
                            leftK * sizeof(float), leftM,
                            tileK * sizeof(float), tileM,
                            a->uram, s->aBase + k * tileK * sizeof(float) + m * tileM * s->aStride + batch * s->aBatchStride);
                    } else {
                        long leftK = s->K - tileK * k;
                        long leftM = s->M - tileM * m;
                        load_matrix<GROUPBASE + LSU_A>(
                            leftK * sizeof(uint16_t), leftM,
                            tileK * sizeof(uint16_t), tileM,
                            a->uram, s->aBase + k * tileK * sizeof(uint16_t) + m * tileM * s->aStride + batch * s->aBatchStride);
                    }
                }
            } else {
                if (n == 0) {
                    // 仅遍历一次，因此仅 m 为当前最后一行时检查
                    // tx_lsu_dma_d2u<GROUPBASE + LSU_A>(a->uram, s->aBase + k * tileK * 4 +
                    //                                            m * tileM * s->aStride);
                    if(A_IS_FP32) {
                        long leftK = s->K - tileK * k;
                        long leftM = s->M - tileM * m;
                        load_matrix<GROUPBASE + LSU_A>(
                            leftK * sizeof(float), leftM,
                            tileK * sizeof(float), tileM,
                            a->uram, s->aBase + k * tileK * sizeof(float) + m * tileM * s->aStride + batch * s->aBatchStride);
                    } else {
                        long leftK = s->K - tileK * k;
                        long leftM = s->M - tileM * m;
                        load_matrix<GROUPBASE + LSU_A>(
                            leftK * sizeof(uint16_t), leftM,
                            tileK * sizeof(uint16_t), tileM,
                            a->uram, s->aBase + k * tileK * sizeof(uint16_t) + m * tileM * s->aStride + batch * s->aBatchStride);
                    }
                }
                if ((m % 2) == 0) {
                    // 多次遍历，因此每次 n 为第一列或者最后一列时检查
                    // tx_lsu_dma_d2u<GROUPBASE + LSU_B>(b->uram, s->bBase + k * tileK * 2 +
                    //                                            n * tileN * s->bStride);
                    long leftK = s->K - tileK * k;
                    long leftN = s->N - tileN * n;
                    if(B_TRANSPOSE) load_matrix<GROUPBASE + LSU_B>(
                        leftK * sizeof(uint16_t), leftN,
                        tileK * sizeof(uint16_t), tileN,
                        b->uram, s->bBase + k * tileK * sizeof(uint16_t) + n * tileN * s->bStride + batch * s->bBatchStride);
                    else load_matrix_transposed<GROUPBASE + LSU_B, 1 /* LOG2(Sizeof(BFloat16)) == 1 */>(
                        leftK, leftN,
                        tileK, tileN,
                        b->uram, s->bBase + k * tileK * s->bStride + n * tileN * sizeof(uint16_t) + batch * s->bBatchStride);
                }
            }
            // tx_lsu_dma_d2u<GROUPBASE + 2>(c->uram,
            //                               s->cBase + n * tileN * 4 + m * tileM * s->cStride);
            
            // 注意：仅当 k 不为第一层切片时，才有必要从内存中加载 C 矩阵。我们这里执行的是 Matmul 而非 MMA。
            if(k) {
                long leftN = s->N - tileN * n;
                long leftM = s->M - tileM * m;
                load_matrix<GROUPBASE + LSU_CLD>(
                    leftN * sizeof(float), leftM,
                    tileN * sizeof(float), tileM,
                    c->uram, s->cBase + n * tileN * sizeof(float) + m * tileM * s->cStride + batch * s->cBatchStride);
            }

            // __tx_lsu_sb_bar();
            // __tx_convert_fp32_bf16_round<LEGACY_A, LEGACY_A>((void *)a->uram, (void *)a->uram);

            NPUDMABar(0);

            if (ZT) {
                if (m == 0) {
                    b->produce(sched);
                }
                if ((n % 2) == 0) {
                    if(A_IS_FP32) {
                        // 可选的转数部分
                        long leftK = s->K - tileK * k;
                        long leftM = s->M - tileM * m;
                        __tx_tcsr_write<LEGACY_A * 8 + TCSR_DIM0>(leftK < tileK ? leftK : tileK);
                        __tx_tcsr_write<LEGACY_A * 8 + TCSR_DIM1>(leftM < tileM ? leftM : tileM);
                        __tx_convert_fp32_bf16_round<LEGACY_A, LEGACY_A>((void *)a->uram, (void *)a->uram);
                        debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: PRODUCER START CONVERT\n",
                            get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
                        NPULEGACYBar(0);
                        debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: PRODUCER FINISH CONVERT\n",
                            get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
                    }
                    a->produce(sched);
                }
            } else {
                if (n == 0) {
                    if(A_IS_FP32) {
                        // 可选的转数部分
                        long leftK = s->K - tileK * k;
                        long leftM = s->M - tileM * m;
                        __tx_tcsr_write<LEGACY_A * 8 + TCSR_DIM0>(leftK < tileK ? leftK : tileK);
                        __tx_tcsr_write<LEGACY_A * 8 + TCSR_DIM1>(leftM < tileM ? leftM : tileM);
                        __tx_convert_fp32_bf16_round<LEGACY_A, LEGACY_A>((void *)a->uram, (void *)a->uram);
                        debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: PRODUCER START CONVERT\n",
                            get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
                        NPULEGACYBar(0);
                        debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: PRODUCER FINISH CONVERT\n",
                            get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
                    }
                    a->produce(sched);
                }
                if ((m % 2) == 0) {
                    b->produce(sched);
                }
            }
            c->produce(sched);
            NPUYield(0); // 让出执行时间，执行其它纤程
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask *p, int *err) {
        return ((MatmulProducerTask *)p)->execute(err);
    }
};

// Consumer Task 类
template <long GROUPBASE, long tileM, long tileN, long tileK, bool ZT, long swizzle, bool A_IS_FP32>
class MatmulConsumerTask : public NPUTask {
  private:
    long cCkSum;
    MatmulPubState<tileM, tileN, tileK, ZT> *s;
    inline void config_tcsr() {
        // C Configuration
        set_lsu_matrix<GROUPBASE + LSU_CST>(cCkSum, tileM, tileN * 4, tileN * 4, s->cStride);
        tx_lsu_tcsr_write<(GROUPBASE + LSU_CST) * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(swizzle);
        // Matmul Configuration
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_M>(tileM - 1);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_N>(tileN - 1);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_K>(tileK - 1);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_ASTRIDE>(tileK * 2); // Stride A
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_BSTRIDE>(tileK * 2); // Stride B
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_CSTRIDE>(tileN * 4); // Stride C
        // tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
        // tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
        // tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_DATATYPE>(0);
        // 暂时关闭 A 上的 Swizzle，以适配向量的转数格式
        // tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(swizzle);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(A_IS_FP32 ? 0 : swizzle);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(swizzle);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(swizzle);
        set_legacy_matrix<LEGACY_A>(tileM, tileK);
    }
    // 局部变量
    NPUURAM *a, *b, *c;
    int m, n, k, batch; // 当前处理中的 m n k batch
    int a_buf_id, b_buf_id, c_buf_id;

  public:
    MatmulConsumerTask(MatmulPubState<tileM, tileN, tileK, ZT> *s) : s(s), NPUTask(wrapper){};
    int32_t execute(int *err) {
        NPUBegin;
        config_tcsr();
        while (1) {
            // 获取一个任务, 注意，m n k 指切片下标
            if (s->advance(m, n, k, batch, a_buf_id, b_buf_id, c_buf_id))
                break;
            debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: CONSUMER GET TASK\n",
                get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
            // 获取并加载 URAM Buffer
            a = &(s->a_buf[a_buf_id]);
            b = &(s->b_buf[b_buf_id]);
            c = &(s->c_buf[c_buf_id]);
            NPUWaitReady(c);
            NPUWaitReady(b);
            NPUWaitReady(a);
            debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: CONSUMER GET READY URAM BUFFER\n",
                get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);

            if(1) {
                long leftM = s->M - tileM * m;
                long leftN = s->N - tileN * n;
                long leftK = s->K - tileK * k;
                ulong real_m = leftM > tileM ? tileM : leftM;
                ulong real_n = leftN > tileN ? tileN : leftN;
                ulong real_k = leftK > tileK ? tileK : leftK;
                tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_M>(real_m - 1);
                tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_N>(real_n - 1);
                tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_K>(real_k - 1);
            }
            // debugf("Calc (%d, %d) -> (%d, %d) = 0x%lx\n", m * tileM, n * tileN,
            //    m * tileM + tileM - 1, n * tileN + tileN - 1, c->uram);
            if(k == 0) tx_mat_mm<GROUPBASE + 0>(c->uram, a->uram, b->uram);
            else tx_mat_mma<GROUPBASE + 0>(c->uram, a->uram, b->uram);

            // 最后一行 / 最后一列 / 第一行 / 第一列时都需要做边界检查
            NPUMatBar(0);
            // 释放 a b
            a->consume(sched);
            b->consume(sched);
            debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: CONSUMER RELEASE URAM BUFFER AND START TO WRITEBACK\n",
                get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
            // 写回 C
            // tx_lsu_dma_u2d<GROUPBASE + 3>(s->cBase + n * tileN * 4 + m * tileM * s->cStride,
            //                               c->uram);
            if(1) {
                long leftM = s->M - tileM * m;
                long leftN = s->N - tileN * n;
                store_matrix<GROUPBASE + LSU_CST>(
                    leftN * sizeof(float), leftM,
                    tileN * sizeof(float), tileM,
                    c->uram, s->cBase + n * tileN * sizeof(float) + m * tileM * s->cStride + batch * s->cBatchStride);
            }
            // Optimize: Early deallocate of C Buffer, since dma instruction is execute inorder.
            NPUDMABar(0);
            c->consume(sched);
            debugf("[%lu]: m(%d), n(%d), k(%d), b(%d), A[%d], B[%d], C[%d]: CONSUMER FINISH WRITEBACK\n",
                get_mcycle, m, n, k, batch, a_buf_id, b_buf_id, c_buf_id);
            // c->consume(sched);
            NPUYield(0);
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask *p, int *err) {
        return ((MatmulConsumerTask *)p)->execute(err);
    }
};
} // namespace

template <long tileM, long tileN, long tileK, bool ZT, bool A_IS_FP32, bool BT>
static inline void matmul_bf16_template(
    uint64_t M, uint64_t N, uint64_t K, uint64_t B,
    uint64_t aBase, uint64_t bBase, uint64_t cBase,
    uint64_t aStride, uint64_t bStride, uint64_t cStride,
    uint64_t aBatchStride, uint64_t bBatchStride, uint64_t cBatchStride
) {
    const ulong a_elem_size = A_IS_FP32 ? sizeof(float) : sizeof(uint16_t);
    NPUURAM buf[7] = {
        NPUURAM(0x0ul),
        NPUURAM(tileM * tileK * a_elem_size),
        NPUURAM(tileM * tileK * a_elem_size * 2),
        NPUURAM(tileM * tileK * a_elem_size * 2 + tileN * tileK * 2),
        NPUURAM(tileM * tileK * a_elem_size * 2 + tileN * tileK * 2 * 2),
        NPUURAM(tileM * tileK * a_elem_size * 2 + tileN * tileK * 2 * 2 + tileM * tileN * 4 * 1),
        NPUURAM(tileM * tileK * a_elem_size * 2 + tileN * tileK * 2 * 2 + tileM * tileN * 4 * 2)};
    debugf("A0[%lx - %lx)\n", buf[0].uram, buf[0].uram + tileM * tileK * a_elem_size);
    debugf("A1[%lx - %lx)\n", buf[1].uram, buf[1].uram + tileM * tileK * a_elem_size);
    debugf("B0[%lx - %lx)\n", buf[2].uram, buf[2].uram + tileN * tileK * 2);
    debugf("B1[%lx - %lx)\n", buf[3].uram, buf[3].uram + tileN * tileK * 2);
    debugf("C0[%lx - %lx)\n", buf[4].uram, buf[4].uram + tileM * tileN * 4);
    debugf("C1[%lx - %lx)\n", buf[5].uram, buf[5].uram + tileM * tileN * 4);
    debugf("C2[%lx - %lx)\n", buf[6].uram, buf[6].uram + tileM * tileN * 4);
    // 计算 M N Tile 的个数，C Buf 不能超过 M N Tile 个数之积
    const uint64_t maxCBuf = ((M + tileM - 1) / tileM) * ((N + tileN - 1) / tileN);
    // 创建两个 PubState，分别表示 Producer 以及 Consumer
    auto pub_producer = MatmulPubState<tileM, tileN, tileK, ZT>(
        M, N, K, B,
        aBase, bBase, cBase,
        aStride, bStride, cStride,
        aBatchStride, bBatchStride, cBatchStride,
        &buf[0], &buf[2], &buf[4], maxCBuf > 3 ? 3 : maxCBuf);
    auto pub_consumer = MatmulPubState<tileM, tileN, tileK, ZT>(pub_producer);

    // 配置 Swizzle 为 M5 B3 S5
    // const long swizzle_pack = MAT_BATCH_SWIZZLE(5, 3, 5);
    const long swizzle_pack = 0;

    // 创建四个 Task
    auto task_producer0 =
        MatmulProducerTask<0, tileM, tileN, tileK, ZT, swizzle_pack, A_IS_FP32, BT>(&pub_producer);
    auto task_producer1 =
        MatmulProducerTask<4, tileM, tileN, tileK, ZT, swizzle_pack, A_IS_FP32, BT>(&pub_producer);
    auto task_consumer0 =
        MatmulConsumerTask<0, tileM, tileN, tileK, ZT, swizzle_pack, A_IS_FP32>(&pub_consumer);
    auto task_consumer1 =
        MatmulConsumerTask<4, tileM, tileN, tileK, ZT, swizzle_pack, A_IS_FP32>(&pub_consumer);
    NPUTask *tasks[4] = {(NPUTask *)&task_producer0, (NPUTask *)&task_producer1,
                         (NPUTask *)&task_consumer0, (NPUTask *)&task_consumer1};
    // 创建一个调度器
    auto sched = NPUScheduler(4, tasks);

    for (auto t : tasks) {
        t->sched = &sched;
    }

    // 开始执行 kernel
    sched.scheduler();
}

template<bool BT>
static inline long matmul_fp32_bf16_fp32(
    uint64_t M, uint64_t N, uint64_t K, uint64_t B,
    uint64_t aBase, uint64_t bBase, uint64_t cBase,
    uint64_t aStride, uint64_t bStride, uint64_t cStride,
    uint64_t aBatchStride, uint64_t bBatchStride, uint64_t cBatchStride) {
    if (M <= N) {
        const uint64_t tileM = 64, tileN = 128, tileK = 512;
        matmul_bf16_template<tileM, tileN, tileK, false, true, BT>(
            M, N, K, B,
            aBase, bBase, cBase,
            aStride, bStride, cStride,
            aBatchStride, bBatchStride, cBatchStride
        );
    } else { // M >= N
        const uint64_t tileM = 64, tileN = 128, tileK = 512;
        matmul_bf16_template<tileM, tileN, tileK, true, true, BT>(
            M, N, K, B,
            aBase, bBase, cBase,
            aStride, bStride, cStride,
            aBatchStride, bBatchStride, cBatchStride
        );
    }
    return 0;
}

// 名称后缀 QK256 是为了兼容之前的算子名称。如要修改算子名称需要同时修改 llama.cpp 和 test_trace/test_matmul.cpp
__device__ long gmma_bf16_fp32_AT_QK256(void *a, void *b, void *c, uint64_t m, uint64_t n, uint64_t k, uint64_t lda,
                                        uint64_t ldb, uint64_t ldc, uint64_t g, uint64_t h, uint64_t stride_g, uint64_t stride_h,
                                        bool transposed) {
    // 在 llama.cpp 中算子接口中的 A 和 B 与算子内部的 A 和 B 相反
    // 如果 ratio * ldb == stride_h，说明 B 布局为 [Head][N][K] ，且紧密排布，可视为 [Ratio*N][K] 直接参与矩阵乘。
    ulong ratio = h / g;
    if(ratio * ldb == stride_h || ratio == 1) {
        // 可直接内联 h 和 n
        // debugf("m(%d), n(%d), ldc(%d)\n", m, n, ldc);
        if(transposed) {
            return matmul_fp32_bf16_fp32<false>(
                (uint64_t)n * ratio, (uint64_t)m, (uint64_t)k, (uint64_t)g,
                (uint64_t)b, (uint64_t)a, (uint64_t)c,
                ldb * sizeof(float), lda * sizeof(uint16_t), ldc * sizeof(float),
                ratio * stride_h * sizeof(float), stride_g * sizeof(uint16_t), ratio * n * ldc * sizeof(float)
            );
        } else {
            return matmul_fp32_bf16_fp32<true>(
                (uint64_t)n * ratio, (uint64_t)m, (uint64_t)k, (uint64_t)g,
                (uint64_t)b, (uint64_t)a, (uint64_t)c,
                ldb * sizeof(float), lda * sizeof(uint16_t), ldc * sizeof(float),
                ratio * stride_h * sizeof(float), stride_g * sizeof(uint16_t), ratio * n * ldc * sizeof(float)
            );
        }
    } else {
        // 暂时不支持离散的情况，这种模式下严重欠优化。
        printf("ERROR[gmma_bf16_fp32_AT_QK256]: h(%d) * ldb(%d) =/= stride_h(%d), g(%d), stride_g(%d)\n", h, ldb, stride_h, g, stride_g);
        return -1;
    }
}