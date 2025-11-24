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

#define NPUYield(z) \
        do { \
            sched->add_free_task(this); \
            _state = &&CONCAT_LINE(_lb_); \
            return z; CONCAT_LINE(_lb_):; \
        } while (0)

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

class NPUScheduler;

// 所有算子的基类
class NPUTask : public NPUListNodeBase {
protected:
    // 基准状态
    void* _state = 0;
    int32_t (*func_ptr)(NPUTask*p, int* err);
public:
    NPUScheduler* sched = 0;
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
    inline void add_free_task(NPUTask* task) {
        free_list.list_insert_tail(task);
    }
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

// 虚拟资源管理类
class NPUResource {
protected:
    enum state_e {
        FREE,
        BUSY,
        READY        
    } state; // 当前虚拟资源状态， 0: 空闲， 1: 未就绪， 2: 就绪
    int producer_count; // 生产者计数器
    int consumer_count; // 消费者计数器
    NPUListHead<NPUTask> wait_free_list; // 等待当前资源 FREE 的 NPUTask 链表
    NPUListHead<NPUTask> wait_ready_list; // 等待当前资源 READY 的 NPUTask 链表
    inline void release_free_list(NPUScheduler* sched) {
        while(!wait_free_list.list_empty()) {
            sched->add_free_task(wait_free_list.list_remove_head());
        }
    }
    inline void release_ready_list(NPUScheduler* sched) {
        while(!wait_ready_list.list_empty()) {
            sched->add_free_task(wait_ready_list.list_remove_head());
        }
    }
public:
    NPUResource() {
        state = FREE;
        producer_count = 0;
        consumer_count = 0;
    }
    inline state_e get_state() {
        return state;
    }
    // 返回 1 时， yield
    inline int wait_ready(NPUTask* cur, NPUScheduler* sched) {
        if(state == READY) {
            // sched->add_free_task(cur);
            return 0;
        } else {
            // FREE or BUSY
            wait_ready_list.list_insert_tail(cur);
            return 1;
        }
    }
    // 返回 1 时， yield
    inline int wait_free(NPUTask* cur, NPUScheduler* sched) {
        if(state == FREE) {
            // sched->add_free_task(cur);
            return 0;
        } else {
            // BUSY or READY
            wait_free_list.list_insert_tail(cur);
            return 1;
        }
    }
    inline int allocate(NPUScheduler* sched, int producer, int consumer) {
        if(state == FREE) {
            // 成功
            producer_count = producer;
            consumer_count = consumer;
            if(producer_count) state = BUSY;
            else {
                release_ready_list(sched);
                state = READY;
            }
            return 0;
        }
        return 1;
    }
    inline void produce(NPUScheduler* sched) {
        if(state != BUSY) {
            printf("Error: produce resource when not busy.\n");
            return;
        }
        producer_count--;
        if(consumer_count == 0) {
            release_ready_list(sched);
            state = READY;
        }
    }
    inline void consume(NPUScheduler* sched) {
        if(state != READY) {
            printf("Error: free resource when not ready.\n");
            return;
        }
        consumer_count--;
        if(consumer_count == 0) {
            release_free_list(sched);
            state = FREE;
        }
    }
};

#define NPUWaitFree(res) \
        do { \
            if(res->wait_free(this, sched)) { \
                _state = &&CONCAT_LINE(_lb_); \
                return 0; CONCAT_LINE(_lb_):; \
            } \
        } while (0)

#define NPUWaitReady(res) \
        do { \
            if(res->wait_ready(this, sched)) { \
                _state = &&CONCAT_LINE(_lb_); \
                return 0; CONCAT_LINE(_lb_):; \
            } \
        } while (0)

class NPUURAM : public NPUResource {
public:
    uint64_t uram;
    NPUURAM(uint64_t uram): uram(uram) {};
};

#endif
