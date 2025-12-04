#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_task.hpp"
#include "npu_vec_csr.hpp"
#include <stdio.h>
#include <string.h>

// Note0：所有 kernel 内部使用的非递归函数建议修饰为 static inline
// 递归函数建议修饰为 static
// 建议仅标记为 __device__ 的函数对外部可见，以避免命名冲突的可能

// Note1：避免过度抽象封装，将重复的/结构性的代码段或者语句组织抽象为函数
// 如若仅出现一次的代码段或者语句，没有必要单独抽象为一个函数

// 基本 calculated goto 测试
static int test_calculated_goto_prototype(void **state) {
    if (*state)
        goto **state;
    static int i;
    for (i = 0; i < 100; i++) {
        *state = &&lb0;
        return i;
    lb0:
        do {
            *state = &&lb3;
            return i * 300;
        lb3:
            break;
        } while (0);
    }
    *state = NULL;
    return 0;
}

__device__ long test_calculated_goto(void) {
    int value = 0;
    void *state = NULL;
    do {
        value = test_calculated_goto_prototype(&state);
        printf("Value: %lx, %x\n", state, value);
    } while (state != NULL);
    return 0;
}

template <long tileSz, long GROUP> class MyTestNPUTask : public NPUTask {
  private:
    char *src, *dst;
    char *uramBuf;
    long size;
    long last_checksum;
    int i;
    void config_tcsr() {
        long cur_checksum;
        if (last_checksum) {
            tx_lsu_tcsr_read<GROUP * LSU_GROUPSZ + LSU_CHECKSUM>(cur_checksum);
            if (last_checksum == cur_checksum) {
                return;
            }
        } else {
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCK>(0);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM0>((tileSz / 64) - 1);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM1>(0);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM2>(0);

            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
            tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTPADDING>(63);

            tx_lsu_tcsr_read<GROUP * LSU_GROUPSZ + LSU_CHECKSUM>(last_checksum);
        }
    }

  public:
    int32_t execute(int *err) {
        NPUBegin;
        for (i = 0; i < (size / tileSz); i++) {
            printf("%dGourp, %d: %d Round\n", GROUP, i, (size / tileSz));
            // 配置张量寄存器
            // 加载数据到 URAM
            config_tcsr();
            tx_lsu_dma_d2u<0>((long)uramBuf, (long)src + i * tileSz);
            NPUDMABar(0);
            // 写回数据到 DRAM
            config_tcsr();
            tx_lsu_dma_u2d<0>((long)dst + i * tileSz, (long)uramBuf);
            NPUDMABar(0);
        }
        NPUFinish(-1);
    }
    // 每一个实现均需要暴露一个静态的 wrapper 函数
    static int32_t wrapper(NPUTask *p, int *err) {
        return ((MyTestNPUTask *)p)->execute(err);
    }
    MyTestNPUTask(char *src, char *dst, char *uramBuf, long size)
        : NPUTask(wrapper), src(src), dst(dst), uramBuf(uramBuf), size(size){};
};

__device__ long test_npu_task_basic_dma(void) {
    static int *inBuf = (int *)0x90000000;
    static int *outBuf = (int *)0x91000000;
    memset(outBuf, 0, 1024 * 1024);
    for (int i = 0; i < 1024 * 1024 / 4; i++) {
        inBuf[i] = i;
    }
    MyTestNPUTask<128 * 1024, 0> task0((char *)inBuf, (char *)outBuf, (char *)0x0, 1024 * 1024 / 2);
    MyTestNPUTask<128 * 1024, 1> task1(((char *)inBuf) + 1024 * 512, ((char *)outBuf) + 1024 * 512,
                                       (char *)(128 * 1024), 1024 * 1024 / 2);
    NPUTask *tasks[2] = {(NPUTask *)&task0, (NPUTask *)&task1};
    NPUScheduler sched(2, tasks);
    task0.sched = &sched;
    task1.sched = &sched;
    sched.scheduler();
    for (int i = 0; i < 1024 * 1024 / 4; i++) {
        if (outBuf[i] != i) {
            printf("Error@%d: %d\n", i, outBuf[i]);
            return 1;
        }
    }
    return 0;
}

// Error@134528: 138624
// 0x83600 -> 0x87600
