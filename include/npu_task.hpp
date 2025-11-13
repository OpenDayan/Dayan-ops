#ifndef _NPU_TASK_HPP_
#define _NPU_TASK_HPP_

#include <cstdint>
#include "npu_instruction.hpp"

// 协程基础宏
#define NPUBegin         switch(_state) { case 0:;
#define NPUFinish(z)     } return (z)

#define NPUDMABar \
        do { \
            tx_lsu_sb_produce((long) this); \
            scrLine=__LINE__; \
            return 0; case __LINE__:; \
        } while (0)

#define NPUMatBar \
        do { \
            tx_mat_sb_produce((long) this); \
            scrLine=__LINE__; \
            return 0; case __LINE__:; \
        } while (0)

#define NPUVecBar \
        do { \
            tx_vec_sb_produce((long) this); \
            scrLine=__LINE__; \
            return 0; case __LINE__:; \
        } while (0)

// 所有算子的基类
class NPUTask {
protected:
    // 基准状态
    uint32_t _state = 0;
public:
    void mybar() {
        long ptr = (long) this;
        tx_lsu_sb_produce(ptr);
    }
    // 协程 Kernel：返回值含义为新创建的协程个数
    virtual int32_t execute() = 0;
};

// 算子调度器类
class NPUScheduler {
protected:
    int32_t taskCounter; // 当前有几个任务正在运行

public:
    // 参数：初始任务个数
    NPUScheduler(int32_t initTaskCount): taskCounter(initTaskCount) {};
    // 阻塞式调度，完成所有调度任务
    // 参数：初始 Task 列表
    int scheduler(NPUTask* initTasks) {
        for(int i = 0; i < taskCounter; i++) {
            taskCounter += initTasks[i].execute();
        }
        // 主循环体
        while(taskCounter) {
            // 获取完成任务的 Task 指针
            long sbInfo;
            tx_sb_get_any(sbInfo);
            if(sbInfo == 0) continue;
            NPUTask* taskPtr = (NPUTask*)sbInfo;
            taskCounter += taskPtr->execute();
        }
        return 0;
    }
};

#endif
