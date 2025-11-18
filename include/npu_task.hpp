#ifndef _NPU_TASK_HPP_
#define _NPU_TASK_HPP_

#include <cstdint>
#include <cstdio>
#include "npu_instruction.hpp"
#include "npu_list.hpp"

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
class NPUTask : public NPUListNodeBase {
protected:
    // 基准状态
    void* _state = 0;
    int32_t (*func_ptr)(NPUTask*p, int* err);
public:
    // 协程 Kernel：返回值含义为新创建的协程个数
    // TODO: 裸机环境下，没有 RTTI，不可使用虚函数。 Fix: 使用手动的虚函数表
    // virtual int32_t execute() = 0;
    int32_t _execute(int* err) {
        if(!func_ptr) {
            printf("Error: Empty NPU Task in %lx.\n", this);
            *err = 1;
            return 0;
        }
        return func_ptr(this, err);
    }
    NPUTask(int32_t (*func_ptr)(NPUTask*p, int* err)): func_ptr(func_ptr) {};
};

// 虚拟资源管理类
class NPUResource {
protected:
    int32_t state;      // 当前虚拟资源状态， 0: 空闲， 1: 未就绪， 2: 就绪
    NPUListHead<NPUTask> wait_list; // 等待当前资源的 NPUTask 链表
};

// 算子调度器类
class NPUScheduler {
protected:
    int32_t taskCounter; // 当前有几个任务正在运行
    NPUListHead<NPUTask> free_list; // 等待当前资源的 NPUTask 链表

public:
    // 参数：初始任务个数
    NPUScheduler(int32_t initTaskCount, NPUTask** initTasks): taskCounter(initTaskCount) {
        for(int i = 0; i < initTaskCount; i ++) {
            free_list.list_insert_tail(initTasks[i]);
        }
    };
    // 阻塞式调度，完成所有调度任务
    // 参数：初始 Task 列表
    int scheduler() {
        int err = 0;
        // 主循环体
        while(taskCounter) {
            if(err) {
                break;
            }
            NPUTask* taskPtr;
            if(!free_list.list_empty()) {
                // 获取 free_list 中的 Task 指针
                taskPtr = free_list.list_remove_head();
            } else {
                // 获取 Scoreboard 中的 Task 指针
                long sbInfo;
                tx_sb_get_any(sbInfo);
                taskPtr = (NPUTask*)sbInfo;
            }
            if(taskPtr == NULL) continue;
            taskCounter += taskPtr->_execute(&err);
        }
        return 0;
    }
};

#endif
