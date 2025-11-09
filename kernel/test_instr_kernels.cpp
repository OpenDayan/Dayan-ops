#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include "npu_vec_csr.hpp"
#include <stdio.h>
#include <string.h>

// Note0：所有 kernel 内部使用的非递归函数建议修饰为 static inline
// 递归函数建议修饰为 static
// 建议仅标记为 __device__ 的函数对外部可见，以避免命名冲突的可能

// Note1：避免过度抽象封装，将重复的/结构性的代码段或者语句组织抽象为函数
// 如若仅出现一次的代码段或者语句，没有必要单独抽象为一个函数

extern int _uram[256 * 1024];

template<long tcsrid>
static inline long test_one_tcsr_reg(void) {
    long mask = 0xffffffffffffffffll;
    long writeMask, clearMask;
    tx_vec_tcsr_write<tcsrid>(0);
    tx_vec_tcsr_write<tcsrid>(mask);
    tx_vec_tcsr_read<tcsrid>(writeMask);
    tx_vec_tcsr_write<tcsrid>(0);
    tx_vec_tcsr_read<tcsrid>(clearMask);
    return writeMask & ~clearMask; // 返回可读写的位掩码
}

__device__ long tx_vec_tcsr_write_read(void) {
    long ret = 0xffffffff;ret &= test_one_tcsr_reg<0>();
    printf("0: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<1>();
    printf("1: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<2>();
    printf("2: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<3>();
    printf("3: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<4>();
    printf("4: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<5>();
    printf("5: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<6>();
    printf("6: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<7>();
    printf("7: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<8>();
    printf("8: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<9>();
    printf("9: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<10>();
    printf("10: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<11>();
    printf("11: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<12>();
    printf("12: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<13>();
    printf("13: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<14>();
    printf("14: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<15>();
    printf("15: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<16>();
    printf("16: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<17>();
    printf("17: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<18>();
    printf("18: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<19>();
    printf("19: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<20>();
    printf("20: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<21>();
    printf("21: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<22>();
    printf("22: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<23>();
    printf("23: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<24>();
    printf("24: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<25>();
    printf("25: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<26>();
    printf("26: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<27>();
    printf("27: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<28>();
    printf("28: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<29>();
    printf("29: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<30>();
    printf("30: %x\n", ret);
    ret = 0xffffffff;ret &= test_one_tcsr_reg<31>();
    printf("31: %x\n", ret);
    return 0;
}

static inline void vec_sb_bar(void) {
    long some_val;
    tx_vec_sb_bar(some_val);
    tx_vec_sb_bar(some_val);
}

__device__ long tx_vec_base_test(void) {
    for(int i = 0 ; i < (16 * 16 * 32); i++) {
        _uram[32 * 256 - 1 - i] = i + 1;
        _uram[32 * 256 + i] = i + 1;
    }
    // 配置寄存器
    tx_vec_tcsr_write<VEC_BLOCK>(31);
    tx_vec_tcsr_write<VEC_DIM0> (15);
    tx_vec_tcsr_write<VEC_DIM1> (15);
    tx_vec_tcsr_write<VEC_GROUPSEL>(0);
    tx_vec_tcsr_write<VEC_DIM0VSTRIDE0> (128);
    tx_vec_tcsr_write<VEC_DIM1VSTRIDE0> (16 * 128);

    // 理论可 INT 满吞吐，LSU 满吞吐（非分离 LSU）
    tx_vec_v_load<0, 0>(0);           // load a
    tx_vec_v_load<1, 0>(32 * 1024);   // load b
    tx_vec_vv_int32_x<0, 1, 17>();    // c = a + b
    tx_vec_vv_int32_x<1, 2, 17>();    // d = a + b
    tx_vec_vv_int32_x<0, 1, 17>();    // e = c + d
    tx_vec_v_store<1, 0>(64 * 1024);  // store e

    // 实测 INT 吞吐： 75%~
    vec_sb_bar();                     // bar
    for(int i = 0 ; i < (16 * 16 * 32); i++) {
        if(_uram[64 * 256 + i] != 16386) {
            printf("dbg: _uram[64 * 256 + %d] == %d\n", i, _uram[64 * 256 + i]);
        }
    }
    return 0;
}
