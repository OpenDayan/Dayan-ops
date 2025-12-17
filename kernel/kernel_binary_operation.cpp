// 双输入单输出类算子核

#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include "npu_lsu_csr.hpp"
#include "npu_mat_csr.hpp"
#include "npu_task.hpp"
#include "npu_vec_csr.hpp"
#include <stdio.h>
#include <string.h>

// Note0：所有 kernel 内部使用的非递归函数建议修饰为 static inline
// 递归函数建议修饰为 static
// 建议仅标记为 __device__ 的函数对外部可见，以避免命名冲突的可能

// Note1：避免过度抽象封装，将重复的/结构性的代码段或者语句组织抽象为函数
// 如若仅出现一次的代码段或者语句，没有必要单独抽象为一个函数

// Producer-Consumer 编程模型

namespace {
const int LSU_A = 0;
const int LSU_B = 1;
const int LSU_C = 2;

// 全局生产者任务板
template <long tileM, long tileN, long tileK, bool ZT> class BinaryPubState {
  public:
    // 任务尺寸
    const int BS, H, N, M;
    // 基地址
    const uint64_t aBase, bBase, cBase;
    NPUURAM *a_buf;
    NPUURAM *b_buf;
    NPUURAM *c_buf;
    BinaryPubState(int BS, int H, int N, int M, uint64_t aBase, uint64_t bBase, uint64_t cBase,
                   NPUURAM *a_buf, NPUURAM *b_buf, NPUURAM *c_buf)
        : BS(BS), H(H), N(N), M(M),
          aBase(aBase), bBase(bBase), cBase(cBase),
          a_buf(a_buf), b_buf(b_buf), c_buf(c_buf){};
    BinaryPubState(const BinaryPubState &other)
        : BS(other.BS), H(other.H), N(other.N), M(other.M),
          aBase(other.aBase), bBase(other.bBase), cBase(other.cBase),
          a_buf(other.a_buf), b_buf(other.b_buf), c_buf(other.c_buf){};

  private:
    // 内部状态
    void *advance_state = 0;
    // 循环变量
    int bs, h, n;

  public:
    // 返回值表示当前任务是否已结束
    inline bool advance() {
        if (advance_state)
            goto *advance_state;
        advance_state = &&Exit;
    Exit:
        return true;
    }
};

} // namespace
