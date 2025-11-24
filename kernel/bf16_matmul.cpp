#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include "npu_vec_csr.hpp"
#include "npu_lsu_csr.hpp"
#include "npu_task.hpp"
#include <stdio.h>
#include <string.h>

// Note0：所有 kernel 内部使用的非递归函数建议修饰为 static inline
// 递归函数建议修饰为 static
// 建议仅标记为 __device__ 的函数对外部可见，以避免命名冲突的可能

// Note1：避免过度抽象封装，将重复的/结构性的代码段或者语句组织抽象为函数
// 如若仅出现一次的代码段或者语句，没有必要单独抽象为一个函数

// Producer-Consumer 编程模型

// 全局生产者任务板
template<long tileM, long tileN, long tileK, bool ZT>
class MatmulPubState {
public:
    // 任务尺寸
    const int M, N, K;
    // 切片个数
    const int cM, cN, cK;
    // 基地址
    const uint64_t aBase, bBase, cBase;
    const uint64_t aStride, bStride, cStride;
    NPUURAM *a_buf;
    NPUURAM *b_buf;
    NPUURAM *c_buf;
    MatmulPubState(
        int M, int N, int K,
        uint64_t aBase, uint64_t bBase, uint64_t cBase,
        uint64_t aStride, uint64_t bStride, uint64_t cStride,
        NPUURAM *a_buf, NPUURAM *b_buf, NPUURAM *c_buf
    ):  M(M), N(N), K(K),
        aBase(aBase), bBase(bBase), cBase(cBase),
        aStride(aStride), bStride(bStride), cStride(cStride),
        a_buf(a_buf), b_buf(b_buf), c_buf(c_buf),
        cM(M / tileM), cN(N / tileN), cK(K / tileK) {};
    MatmulPubState(const MatmulPubState& other)
        : M(other.M), N(other.N), K(other.K),
          cM(other.cM), cN(other.cN), cK(other.cK),
          aBase(other.aBase), bBase(other.bBase), cBase(other.cBase),
          aStride(other.aStride), bStride(other.bStride), cStride(other.cStride)
    {}
private:
    // 内部状态
    void* advance_state = 0;
    // 循环变量
    int block_row, block_col, m, n, k;
public:
    // 返回值表示当前任务是否已结束
    inline bool advance(int &cur_m, int &cur_n, int &cur_k) {
        if(advance_state) goto *advance_state;
        for(k = 0; k < cK; k++) {
            if(ZT) {
                // 列 -> 行 遍历顺序
                for(block_col = 0; block_col < cN; block_col += 2) {
                    for(block_row = 0; block_row < cM; block_row += 2) {
                        for(m = block_row; (m < cM) && (m < block_row + 2); m ++) {
                            for(n = block_col; (n < cN) && (n < block_col + 2); n ++) {
                                cur_m = m;
                                cur_n = n;
                                cur_k = k;
                                advance_state = &&ZTNextLoop;
                                return false; ZTNextLoop:;
                            }
                        }
                    }
                }
            } else {
                // 行 -> 列 遍历顺序
                for(block_row = 0; block_row < cM; block_row += 2) {
                    for(block_col = 0; block_col < cN; block_col += 2) {
                        for(m = block_row; (m < cM) && (m < block_row + 2); m ++) {
                            for(n = block_col; (n < cN) && (n < block_col + 2); n ++) {
                                cur_m = m;
                                cur_n = n;
                                cur_k = k;
                                advance_state = &&ZNextLoop;
                                return false; ZNextLoop:;
                            }
                        }
                    }
                }
            }
        }
        advance_state = &&Exit;
        Exit: return true;
    }
};

// Producer Task 类
template<long GROUPBASE, long tileM, long tileN, long tileK, bool ZT>
class MatmulProducerTask: public NPUTask {
private:
    int m, n, k; // 当前处理中的 m n k
    MatmulPubState<tileM, tileN, tileK, ZT> *s;
    inline void config_tcsr() {
        // Note: k and tileK should be aligned to 64-Bytes
        // 暂不考虑 K pad 问题。
        tx_lsu_tcsr_write<(GROUPBASE + 0) * LSU_GROUPSZ + LSU_BLOCK>(0);
        tx_lsu_tcsr_write<(GROUPBASE + 0) * LSU_GROUPSZ + LSU_DIM0>(tileK / 32); // tileK * 2 / 64B
        tx_lsu_tcsr_write<(GROUPBASE + 0) * LSU_GROUPSZ + LSU_DIM1>(tileM);      // tileM
        tx_lsu_tcsr_write<(GROUPBASE + 0) * LSU_GROUPSZ + LSU_DIM2>(0);
        tx_lsu_tcsr_write<(GROUPBASE + 0) * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);

    }
public:
    MatmulProducerTask(MatmulPubState<tileM, tileN, tileK, ZT> *s):
        s(s), NPUTask(wrapper) {};
    int32_t execute(int* err) {
        NPUBegin;
        while(1) {
            // 获取任务
            if(s->advance(m, n, k)) break;
            // 获取并加载 URAM Buffer
            NPUURAM *a, *b, *c;
            if(ZT) {
                if(m == 0) {
                    // 负责加载 B Buffer
                    int b_buf_id = n + k * s->cN; b = &(s->b_buf[b_buf_id & 0x1]);
                    do { NPUWaitFree(b); } while(b->allocate(sched, 1, s->cM));
                }
                int a_buf_id = m + k * s->cM; a = &(s->a_buf[a_buf_id & 0x1]);
                do { NPUWaitFree(a); } while(a->allocate(sched, 1, 1));
            } else {
                if(n == 0) {
                    // 负责加载 A Buffer
                    int a_buf_id = m + k * s->cM; a = &(s->a_buf[a_buf_id & 0x1]);
                    do { NPUWaitFree(a); } while(a->allocate(sched, 1, s->cN));
                }
                int b_buf_id = n + k * s->cN; b = &(s->b_buf[b_buf_id & 0x1]);
                do { NPUWaitFree(b); } while(b->allocate(sched, 1, 1));
            }
            int c_buf_id = n + m * s->cN; c = &(s->c_buf[c_buf_id & 0x1]);
            do { NPUWaitFree(c); } while(c->allocate(sched, 1, 1));
            config_tcsr();
            if(ZT) {
                if(m == 0) {
                    tx_lsu_dma_d2u<GROUPBASE + 1>(b->uram, s->bBase + k * tileK * 2 + n * tileN * s->bStride);
                }
                tx_lsu_dma_d2u<GROUPBASE + 0>(a->uram, s->aBase + k * tileK * 2 + m * tileM * s->aStride);
            }
            else {
                if(n == 0) {
                    tx_lsu_dma_d2u<GROUPBASE + 0>(a->uram, s->aBase + k * tileK * 2 + m * tileM * s->aStride);
                }
                tx_lsu_dma_d2u<GROUPBASE + 1>(b->uram, s->bBase + k * tileK * 2 + n * tileN * s->bStride);
            }
            tx_lsu_dma_d2u<GROUPBASE + 2>(c->uram, s->cBase + n * tileN * 4 + m * tileM * s->cStride);
            NPUDMABar(0);
            if(ZT) {
                if(m == 0) {
                    b->produce(sched);
                }
                a->produce(sched);
            }
            else {
                if(n == 0) {
                    a->produce(sched);
                }
                b->produce(sched);
            }
            c->produce(sched);
            NPUYield(0); // 让出执行时间，执行其它纤程
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask*p, int* err) {
        return ((MatmulProducerTask*)p)->execute(err);
    }
};

// Consumer Task 类
template<long GROUPBASE, long tileM, long tileN, long tileK, bool ZT>
class MatmulConsumerTask: public NPUTask {
private:
    int m, n, k; // 当前处理中的 m n k
    MatmulPubState<tileM, tileN, tileK, ZT> *s;
    inline void config_tcsr() {
    }
public:
    MatmulConsumerTask(MatmulPubState<tileM, tileN, tileK, ZT> *s):
        s(s), NPUTask(wrapper) {};
    int32_t execute(int* err) {
        NPUBegin;
        while(1) {
            // 获取一个任务, 注意，m n k 指切片下标
            if(s->advance(m, n, k)) break;
            // 获取并加载 URAM Buffer
            NPUURAM *a, *b, *c;
            int a_buf_id = m + k * s->cM;
            a = &(s->a_buf[a_buf_id & 0x1]);
            int b_buf_id = n + k * s->cN;
            b = &(s->b_buf[b_buf_id & 0x1]);
            int c_buf_id = n + m * s->cN;
            c = &(s->c_buf[c_buf_id & 0x1]);
            NPUWaitReady(c);
            NPUWaitReady(b);
            NPUWaitReady(a);

            config_tcsr();
            tx_mat_mma<GROUPBASE + 0>(c->uram, a->uram, b->uram);

            NPUMatBar(0);
            // 释放 a b
            a->consume(sched);
            b->consume(sched);
            // 写回 C
            tx_lsu_dma_u2d<GROUPBASE + 3>(s->cBase + n * tileN * 4 + m * tileM * s->cStride, c->uram);
            NPUDMABar(0);
            c->consume(sched);
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask*p, int* err) {
        return ((MatmulConsumerTask*)p)->execute(err);
    }
};

__device__ long matmul_bf16_a_bt_c(
    uint64_t M, uint64_t N, uint64_t K,
    uint64_t aBase, uint64_t bBase, uint64_t cBase,
    uint64_t aStride, uint64_t bStride, uint64_t cStride
) {
    const uint64_t tileM = 128, tileN = 128, tileK = 1024;
    NPUURAM buf[6] = {
        NPUURAM(0x0ul),
        NPUURAM(tileM * tileK * 2),
        NPUURAM(tileM * tileK * 2 * 2),
        NPUURAM(tileM * tileK * 2 * 2 + tileN * tileK * 2),
        NPUURAM(tileM * tileK * 2 * 2 + tileN * tileK * 4),
        NPUURAM(tileM * tileK * 2 * 2 + tileN * tileK * 4 + tileM * tileN * 4)
    };
    // 创建两个 PubState，分别表示 Producer 以及 Consumer
    auto pub_producer = MatmulPubState<tileM, tileN, tileK, false>(
        M, N, K, aBase, bBase, cBase, aStride, bStride, cStride,
        &buf[0], &buf[2], &buf[4]
    );
    auto pub_consumer = MatmulPubState<tileM, tileN, tileK, false>(pub_producer);

    // 创建四个 Task
    auto task_producer0 = MatmulProducerTask<0, tileM, tileN, tileK, false> (&pub_producer);
    auto task_producer1 = MatmulProducerTask<4, tileM, tileN, tileK, false> (&pub_producer);
    auto task_consumer0 = MatmulConsumerTask<0, tileM, tileN, tileK, false> (&pub_consumer);
    auto task_consumer1 = MatmulConsumerTask<4, tileM, tileN, tileK, false> (&pub_consumer);
    NPUTask *tasks[4] = {
        (NPUTask*) &task_producer0, (NPUTask*) &task_producer1,
        (NPUTask*) &task_consumer0, (NPUTask*) &task_consumer1
    };

    // 创建一个调度器
    auto sched = NPUScheduler(4, tasks);

    // 开始执行 kernel
    sched.scheduler();

    return 0;
} 
