#ifndef _NPU_TASK_HPP_
#define _NPU_TASK_HPP_

#include <cstdint>
#include <cstdio>
#include "npu_instruction.hpp"

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT_LINE(prefix) CONCAT(prefix, __LINE__)

// 协程基础宏
#define NPUBegin         if(_state) goto *_state;
#define NPUFinish(z)     return (z)

#define NPUDMABar(z) \
        do { \
            tx_lsu_sb_produce((long) this); \
            _state = &&CONCAT_LINE(_lb_); \
            return z; CONCAT_LINE(_lb_):; \
        } while (0)

#define NPUMatBar(z) \
        do { \
            tx_mat_sb_produce((long) this); \
            _state = &&CONCAT_LINE(_lb_); \
            return z; CONCAT_LINE(_lb_):; \
        } while (0)

#define NPUVecBar(z) \
        do { \
            tx_vec_sb_produce((long) this); \
            _state = &&CONCAT_LINE(_lb_); \
            return z; CONCAT_LINE(_lb_):; \
        } while (0)

// 所有算子的基类
class NPUTask {
protected:
    // 基准状态
    void* _state = 0;
    int32_t (*func_ptr)(NPUTask*p, int* err) = 0;
public:
    // 协程 Kernel：返回值含义为新创建的协程个数
    // TODO: 裸机环境下，没有 RTTI，不可使用虚函数。 Fix: 使用手动的虚函数表
    // virtual int32_t execute() = 0;
    int32_t _execute(int* err) {
        if(!func_ptr) {
            printf("Error: Empty NPU Task.\n");
            *err = 1;
            return 0;
        }
        return func_ptr(this, err);
    }
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
    int scheduler(NPUTask** initTasks) {
        int err = 0;
        for(int i = 0; i < taskCounter; i++) {
            taskCounter += initTasks[i]->_execute(&err);
        }
        // 主循环体
        while(taskCounter) {
            if(err) {
                break;
            }
            // 获取完成任务的 Task 指针
            long sbInfo;
            tx_sb_get_any(sbInfo);
            if(sbInfo == 0) continue;
            NPUTask* taskPtr = (NPUTask*)sbInfo;
            taskCounter += taskPtr->_execute(&err);
        }
        return 0;
    }
};

#endif
