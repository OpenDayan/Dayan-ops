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

// 注意：简化版算子，仅考虑已对齐情况。

// Producer-Consumer 编程模型测试

// 全局生产者任务板
struct MatmulProducerPubState {
    int m, n, k;
};

// Producer Task 类
template<long GROUPBASE>
class MatmulProducerTask: public NPUTask {
private:
    // 分块尺寸
    const int tileM, tileN, tileK;
    // 任务尺寸
    const int M, N, K;
    // 切片个数
    const int cM, cN, cK;
    // 基地址
    const uint64_t aBase, bBase, cBase;
    int m, n, k; // 当前处理中的 m n k
    MatmulProducerPubState *s;
    NPUURAM *a_buf[2];
    NPUURAM *b_buf[2];
    NPUURAM *c_buf[2];
    inline void config_tcsr() {
    }
public:
    int32_t execute(int* err) {
        NPUBegin;
        while(1) {
            // 获取一个任务, 注意，m n k 指切片下标
            m = s->m; n = s->n; k = s->k;
            if(k == cK) break;
            // 更新全局任务组
            if((++s->n) == cN) {
                s->n = 0;
                if((++s->m) == cM) {
                    s->m = 0;
                    if((++s->k) == cK) {
                        break;
                    }
                }
            }
            // 获取并加载 URAM Buffer
            NPUURAM *a, *b, *c;
            if(n == 0) {
                // 负责加载 A Buffer
                int a_buf_id = m + k * cM;
                a = a_buf[a_buf_id & 0x1];
                do {
                    NPUWaitFree(a);
                } while(a->allocate(sched, 1, cN));
            }
            int b_buf_id = n + k * cN;
            b = b_buf[b_buf_id & 0x1];
            do {
                NPUWaitFree(b);
            } while(b->allocate(sched, 1, 1));
            
            int c_buf_id = n + m * cN;
            c = c_buf[c_buf_id & 0x1];
            do {
                NPUWaitFree(c);
            } while(c->allocate(sched, 1, 1));

            config_tcsr();
            if(n == 0) tx_lsu_dma_d2u<GROUPBASE + 0>(a->uram, aBase + (m * tileM * K + k * tileK) * 2);
            tx_lsu_dma_d2u<GROUPBASE + 1>(b->uram, bBase + (n * tileN * K + k * tileK) * 2);
            tx_lsu_dma_d2u<GROUPBASE + 2>(c->uram, cBase + (m * tileM * N + n * tileN) * 4);

            NPUDMABar(0);
            if(n == 0) a->produce(sched);
            b->produce(sched);
            c->produce(sched);
            NPUYield(0); // 让出执行时间，执行其它纤程
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask*p, int* err) {
        return ((MatmulProducerTask*)p)->execute(err);
    }
    MatmulProducerTask():
        NPUTask(wrapper) {};
};

// 全局消费者任务板
struct MatmulConsumerPubState {
    int m, n, k;
};

// Consumer Task 类
template<long GROUPBASE>
class MatmulConsumerTask: public NPUTask {
private:
    // 分块尺寸
    const int tileM, tileN, tileK;
    // 任务尺寸
    const int M, N, K;
    // 切片个数
    const int cM, cN, cK;
    // 基地址
    const uint64_t aBase, bBase, cBase;
    int m, n, k; // 当前处理中的 m n k
    MatmulConsumerPubState *s;
    NPUURAM *a_buf[2];
    NPUURAM *b_buf[2];
    NPUURAM *c_buf[2];
    inline void config_tcsr() {
    }
public:
    int32_t execute(int* err) {
        NPUBegin;
        while(1) {
            // 获取一个任务, 注意，m n k 指切片下标
            m = s->m; n = s->n; k = s->k;
            if(k == cK) break;
            // 更新全局任务组
            if((++s->n) == cN) {
                s->n = 0;
                if((++s->m) == cM) {
                    s->m = 0;
                    if((++s->k) == cK) {
                        break;
                    }
                }
            }
            // 获取并加载 URAM Buffer
            NPUURAM *a, *b, *c;
            int a_buf_id = m + k * cM;
            a = a_buf[a_buf_id & 0x1];
            int b_buf_id = n + k * cN;
            b = b_buf[b_buf_id & 0x1];
            int c_buf_id = n + m * cN;
            c = c_buf[c_buf_id & 0x1];
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
            tx_lsu_dma_u2d<GROUPBASE + 3>(cBase + (m * tileM * N + n * tileN) * 4, c->uram);
            NPUDMABar(0);
            c->consume(sched);
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask*p, int* err) {
        return ((MatmulConsumerTask*)p)->execute(err);
    }
    MatmulConsumerTask():
        NPUTask(wrapper) {};
};
