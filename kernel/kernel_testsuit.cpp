#include "npu_kernel.h"

__device__ int kernel_testsuit(
    long a0, long a1, long a2, long a3, long a4, long a5, long a6, long a7,
    long a8, long a9, long a10, long a11, long a12, long a13, long a14, long a15,
    long a16, long a17, long a18, long a19, long a20, long a21, long a22, long a23,
    long a24, long a25, long a26, long a27, long a28, long a29
) {
    volatile long *ptr = (long *)0x81000000;
    ptr[0] = a0;
    ptr[1] = a0 | a1;
    ptr[2] = a0 | a2;
    ptr[3] = a0 | a3;
    ptr[4] = a0 | a4;
    ptr[5] = a0 | a5;
    ptr[6] = a0 | a6;
    ptr[7] = a0 | a7;
    ptr[8] = a0 | a8;
    ptr[9] = a0 | a9;
    ptr[10] = a0 | a10;
    ptr[11] = a0 | a11;
    ptr[12] = a0 | a12;
    ptr[13] = a0 | a13;
    ptr[14] = a0 | a14;
    ptr[15] = a0 | a15;
    ptr[16] = a0 | a16;
    ptr[17] = a0 | a17;
    ptr[18] = a0 | a18;
    ptr[19] = a0 | a19;
    ptr[20] = a0 | a20;
    ptr[21] = a0 | a21;
    ptr[22] = a0 | a22;
    ptr[23] = a0 | a23;
    ptr[24] = a0 | a24;
    ptr[25] = a0 | a25;
    ptr[26] = a0 | a26;
    ptr[27] = a0 | a27;
    ptr[28] = a0 | a28;
    ptr[29] = a0 | a29;
    // ptr = (long*)0x81000100;
    // while((long)ptr < 0x82000000) {
    //     *ptr = a0 + (long)ptr;
    //     ptr++;
    // }

    return a0 + 1;
}

__device__ int kernel_trap_testsuit() {
    volatile long *ptr = (long *)0x0;

    return *ptr;
}