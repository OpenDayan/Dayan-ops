#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

// Note0：所有 kernel 内部使用的非递归函数建议修饰为 static inline
// 递归函数建议修饰为 static
// 建议仅标记为 __device__ 的函数对外部可见，以避免命名冲突的可能

// Note1：避免过度抽象封装，将重复的/结构性的代码段或者语句组织抽象为函数
// 如若仅出现一次的代码段或者语句，没有必要单独抽象为一个函数

template<long unit, long tcsrid>
static inline long test_one_tcsr_reg(void) {
    long mask = 0xffffffffffffffffll;
    long writeMask, clearMask;
    __tx_unit_tcsr_write<unit, tcsrid>(0);
    __tx_unit_tcsr_write<unit, tcsrid>(mask);
    __tx_unit_tcsr_read<unit, tcsrid>(writeMask);
    __tx_unit_tcsr_write<unit, tcsrid>(0);
    __tx_unit_tcsr_read<unit, tcsrid>(clearMask);
    return writeMask & ~clearMask; // 返回可读写的位掩码
}

static inline long test_load_add_store(void *addr) {
    __tx_vec_v_load<0>(addr);
    __tx_vec_v_load<0>(addr);
    __tx_vec_vv_int32_add<0, 1, 0>();
    __tx_vec_v_store<2>(addr);
    return 0;
}

__device__ long tx_unit_tcsr_write_read(void){
    long ret = 0xfffff; // 至少有 20 位可读写位
    ret &= test_one_tcsr_reg<UNIT_MAT, 0>();
    ret &= test_one_tcsr_reg<UNIT_MAT, 1>();
    ret &= test_one_tcsr_reg<UNIT_MAT, 2>();
    test_load_add_store(0);
    return ret == 0xfffff ? 0 : -1;
}

