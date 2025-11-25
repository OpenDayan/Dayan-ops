#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include "npu_vec_csr.hpp"
#include "npu_lsu_csr.hpp"
#include "npu_mat_csr.hpp"
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
          aBase(other.aBase), bBase(other.bBase), cBase(other.cBase),
          aStride(other.aStride), bStride(other.bStride), cStride(other.cStride),
          a_buf(other.a_buf), b_buf(other.b_buf), c_buf(other.c_buf),
          cM(other.cM), cN(other.cN), cK(other.cK) {};
private:
    // 内部状态
    void* advance_state = 0;
    // 循环变量
    int block_row, block_col, m, n, k, rst_id, cur_id, cid;
public:
    // 返回值表示当前任务是否已结束
    inline bool advance(int &cur_m, int &cur_n, int &cur_k, int &a_buf_id, int &b_buf_id, int &c_buf_id) {
        if(advance_state) goto *advance_state;
        cur_id = 0; cid = 0;
        for(k = 0; k < cK; k++) {
            if(ZT) {
                // 列 -> 行 遍历顺序
                for(block_col = 0; block_col < cN; block_col += 2) {
                    rst_id = cur_id;
                    for(block_row = 0; block_row < cM; block_row += 2) {
                        for(m = block_row; (m < cM) && (m < block_row + 2); m ++) {
                            for(n = block_col; (n < cN) && (n < block_col + 2); n ++) {
                                cur_m = m;
                                cur_n = n;
                                cur_k = k;
                                a_buf_id = cur_id;
                                b_buf_id = n & 0x1;
                                c_buf_id = cid;
                                cid ^= 0x1;
                                advance_state = &&ZTNextLoop;
                                return false; ZTNextLoop:;
                            }
                            cur_id ^= 1;
                        }
                        cur_id = rst_id;
                    }
                    cur_id ^= 1;
                }
            } else {
                // 行 -> 列 遍历顺序
                for(block_row = 0; block_row < cM; block_row += 2) {
                    rst_id = cur_id;
                    for(block_col = 0; block_col < cN; block_col += 2) {
                        for(m = block_row; (m < cM) && (m < block_row + 2); m ++) {
                            for(n = block_col; (n < cN) && (n < block_col + 2); n ++) {
                                cur_m = m;
                                cur_n = n;
                                cur_k = k;
                                a_buf_id = m & 0x1;
                                b_buf_id = cur_id;
                                c_buf_id = cid;
                                cur_id ^= 0x1;
                                cid ^= 0x1;
                                advance_state = &&ZNextLoop;
                                return false; ZNextLoop:;
                            }
                            cur_id = rst_id;
                        }
                    }
                    cur_id ^= 0x1;
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
    long aCkSum, bCkSum, cCkSum;// tcsr 配置 cksum
    MatmulPubState<tileM, tileN, tileK, ZT> *s;
    inline void config_tcsr() {
        // Note: k and tileK should be aligned to 64-Bytes
        // 暂不考虑 K pad 问题。
        // A Configuration
        set_lsu_matrix<GROUPBASE + 0>(aCkSum, tileM, tileK * 2, s->aStride, tileK * 2);
        // B Configuration
        set_lsu_matrix<GROUPBASE + 1>(bCkSum, tileN, tileK * 2, s->bStride, tileK * 2);
        // C Configuration
        set_lsu_matrix<GROUPBASE + 2>(cCkSum, tileM, tileN * 4, s->cStride, tileN * 4);
        return;
    }
    // 局部变量
    int m, n, k; // 当前处理中的 m n k
    NPUURAM *a, *b, *c;
    int a_buf_id, b_buf_id, c_buf_id;
public:
    MatmulProducerTask(MatmulPubState<tileM, tileN, tileK, ZT> *s):
        s(s), NPUTask(wrapper) {};
    int32_t execute(int* err) {
        NPUBegin;
        config_tcsr();
        while(1) {
            // 获取任务
            if(s->advance(m, n, k, a_buf_id, b_buf_id, c_buf_id)) break;
            // 获取并加载 URAM Buffer
            c = &(s->c_buf[c_buf_id]);
            do { NPUWaitFree(c); } while(c->allocate(sched, 1, 1));
            if(ZT) {
                if((n % 2) == 0) {
                    a = &(s->a_buf[a_buf_id]);
                    do { NPUWaitFree(a); } while(a->allocate(sched, 1, (s->cN - n) > 2 ? 2 : (s->cN - n)));
                }
                if(m == 0) {
                    // 负责加载 B Buffer
                    b = &(s->b_buf[b_buf_id]);
                    do { NPUWaitFree(b); } while(b->allocate(sched, 1, s->cM));
                }
            } else {
                if((m % 2) == 0) {
                    b = &(s->b_buf[b_buf_id]);
                    do { NPUWaitFree(b); } while(b->allocate(sched, 1, (s->cM - m) > 2 ? 2 : (s->cM - m)));
                }
                if(n == 0) {
                    // 负责加载 A Buffer
                    a = &(s->a_buf[a_buf_id]);
                    do { NPUWaitFree(a); } while(a->allocate(sched, 1, s->cN));
                }
            }
            if(ZT) {
                if(m == 0) {
                    tx_lsu_dma_d2u<GROUPBASE + 1>(b->uram, s->bBase + k * tileK * 2 + n * tileN * s->bStride);
                }
                if((n % 2) == 0) {
                    tx_lsu_dma_d2u<GROUPBASE + 0>(a->uram, s->aBase + k * tileK * 2 + m * tileM * s->aStride);
                }
            }
            else {
                if(n == 0) {
                    tx_lsu_dma_d2u<GROUPBASE + 0>(a->uram, s->aBase + k * tileK * 2 + m * tileM * s->aStride);
                }
                if((m % 2) == 0) {
                    tx_lsu_dma_d2u<GROUPBASE + 1>(b->uram, s->bBase + k * tileK * 2 + n * tileN * s->bStride);
                }
            }
            tx_lsu_dma_d2u<GROUPBASE + 2>(c->uram, s->cBase + n * tileN * 4 + m * tileM * s->cStride);
            NPUDMABar(0);
            if(ZT) {
                if(m == 0) {
                    b->produce(sched);
                }
                if((n % 2) == 0) {
                    a->produce(sched);
                }
            }
            else {
                if(n == 0) {
                    a->produce(sched);
                }
                if((m % 2) == 0) {
                    b->produce(sched);
                }
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
    long cCkSum;
    MatmulPubState<tileM, tileN, tileK, ZT> *s;
    inline void config_tcsr() {
        // C Configuration
        set_lsu_matrix<GROUPBASE + 3>(cCkSum, tileM, tileN * 4, tileN * 4, s->cStride);
        // Matmul Configuration
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_M>(tileM - 1);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_N>(tileN - 1);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_K>(tileK - 1);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_ASTRIDE>(tileK * 2); // Stride A
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_BSTRIDE>(tileK * 2); // Stride B
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_CSTRIDE>(tileN * 4); // Stride C
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_ASWIZZLEMBS>(0);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_BSWIZZLEMBS>(0);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_CSWIZZLEMBS>(0);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_PERSISTENTMODE>(0);
        tx_mat_tcsr_write<GROUPBASE * MAT_GROUPSZ + MAT_DATATYPE>(0);
    }
    // 局部变量
    NPUURAM *a, *b, *c;
    int m, n, k; // 当前处理中的 m n k
    int a_buf_id, b_buf_id, c_buf_id;
public:
    MatmulConsumerTask(MatmulPubState<tileM, tileN, tileK, ZT> *s):
        s(s), NPUTask(wrapper) {};
    int32_t execute(int* err) {
        NPUBegin;
        config_tcsr();
        while(1) {
            // 获取一个任务, 注意，m n k 指切片下标
            if(s->advance(m, n, k, a_buf_id, b_buf_id, c_buf_id)) break;
            // 获取并加载 URAM Buffer
            a = &(s->a_buf[a_buf_id]);
            b = &(s->b_buf[b_buf_id]);
            c = &(s->c_buf[c_buf_id]);
            NPUWaitReady(c);
            NPUWaitReady(b);
            NPUWaitReady(a);

            printf("Calc (%d, %d) -> (%d, %d) = 0x%lx\n",
                m * tileM, n * tileN, m * tileM + tileM - 1, n * tileN + tileN - 1, c->uram);
            tx_mat_mma<GROUPBASE + 0>(c->uram, a->uram, b->uram);

            NPUMatBar(0);
            // 释放 a b
            a->consume(sched);
            b->consume(sched);
            // 写回 C
            tx_lsu_dma_u2d<GROUPBASE + 3>(s->cBase + n * tileN * 4 + m * tileM * s->cStride, c->uram);
            NPUDMABar(0);
            c->consume(sched);
            NPUYield(0);
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
    const uint64_t tileM = 128, tileN = 128, tileK = 512;
    NPUURAM buf[6] = {
        NPUURAM(0x0ul),             // 0    - 128K
        NPUURAM(tileM * tileK * 2), // 128K - 256K
        NPUURAM(tileM * tileK * 2 * 2),                     // 256K - 384K
        NPUURAM(tileM * tileK * 2 * 2 + tileN * tileK * 2), // 384K - 512K
        NPUURAM(tileM * tileK * 2 * 2 + tileN * tileK * 2 * 2),                    // 512K - 576K
        NPUURAM(tileM * tileK * 2 * 2 + tileN * tileK * 2 * 2 + tileM * tileN * 4) // 576K - 640K
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

    for(auto t : tasks) {
        t->sched = &sched;
    }

    // 开始执行 kernel
    sched.scheduler();

    return 0;
} 
