#include "npu_instruction.hpp"
#include <stdio.h>

#define check(x, expect) \
    if ((x) != (expect)) { \
        printf("Error: %s:%d: %s = %ld, expected %ld\n", __FILE__, __LINE__, #x, (x), (expect)); \
        return -1; \
    }

extern "C" long test_tcsr_wr() {
    printf("test_tcsr_wr\n");
    // Setup TCSR
    long old_val;
    __tx_tcsr_write<0>(1);
    __tx_tcsr_write<1>(2);
    __tx_tcsr_write<2>(3);
    __tx_tcsr_write<3>(4);
    __tx_tcsr_xchg<0>(1+4, old_val);
    check(old_val, 1);
    __tx_tcsr_xchg<1>(2+4, old_val);
    check(old_val, 2);
    __tx_tcsr_xchg<2>(3+4, old_val);
    check(old_val, 3);
    __tx_tcsr_xchg<3>(4+4, old_val);
    check(old_val, 4);
    __tx_tcsr_xchg<0>(1+8, old_val);
    check(old_val, 5);
    __tx_tcsr_xchg<1>(2+8, old_val);
    check(old_val, 6);
    __tx_tcsr_xchg<2>(3+8, old_val);
    check(old_val, 7);
    __tx_tcsr_xchg<3>(4+8, old_val);
    check(old_val, 8);
    __tx_tcsr_xchg<0>(0, old_val);
    check(old_val, 9);
    __tx_tcsr_xchg<1>(0, old_val);
    check(old_val, 10);
    __tx_tcsr_xchg<2>(0, old_val);
    check(old_val, 11);
    __tx_tcsr_xchg<3>(0, old_val);
    check(old_val, 12);
    printf("test_tcsr_wr success\n");
    return 0;
}

extern "C" long test_tx_load_store(char* buffer, long buffer_size, long buffer_size_2d0, long buffer_size_2d1) {
    __tx_tcsr_write<TCSR_DIM2>(1);
    __tx_tcsr_write<TCSR_DIM3>(1);
    check(buffer_size_2d0 * buffer_size_2d1, buffer_size);
    printf("test_tx_load_store @%p with %d Bytes size\n", buffer, buffer_size);
    // Update TCSR
    // Generate Load Datas
    for(int i = 0; i < buffer_size; i++) {
        buffer[i] = (char)i;
        buffer[buffer_size + i] = -1;
    }
    // Plain Load/Store
    __tx_tcsr_write<TCSR_DIM0>(buffer_size);
    __tx_tcsr_write<TCSR_STRIDE0>(1);
    __tx_tcsr_write<TCSR_DIM1>(1);
    __tx_tcsr_write<TCSR_STRIDE1>(buffer_size);
    printf("Plain 1D Load/Store begin\n");
    __tx_async_uram_load<0, 0, 0>(buffer, (void*)0);
    __tx_fence();
    __tx_async_uram_store<0, 0, 0>((void*)0, buffer + buffer_size);
    __tx_fence();
    for(int i = 0; i < buffer_size; i++) {
        check(buffer[buffer_size + i], (char)i);
        buffer[buffer_size + i] = -1;
    }

    // Plain Load/Store, Transposed
    __tx_tcsr_write<TCSR_DIM0>(1);
    __tx_tcsr_write<TCSR_STRIDE0>(buffer_size);
    __tx_tcsr_write<TCSR_DIM1>(buffer_size);
    __tx_tcsr_write<TCSR_STRIDE1>(1);
    printf("Plain 1D Load/Store(Transposed) begin\n");
    __tx_async_uram_load<0, 0, 0>(buffer, (void*)0);
    __tx_fence();
    __tx_async_uram_store<0, 0, 0>((void*)0, buffer + buffer_size);
    __tx_fence();
    for(int i = 0; i < buffer_size; i++) {
        check(buffer[buffer_size + i], (char)i);
        buffer[buffer_size + i] = -1;
    }

    // 2D Load/Store
    __tx_tcsr_write<TCSR_DIM0>(buffer_size_2d0);
    __tx_tcsr_write<TCSR_STRIDE0>(1);
    __tx_tcsr_write<TCSR_DIM1>(buffer_size_2d1);
    __tx_tcsr_write<TCSR_STRIDE1>(buffer_size_2d0);
    printf("Plain 2D Load/Store begin\n");
    __tx_async_uram_load<0, 0, 0>(buffer, (void*)0);
    __tx_fence();
    __tx_async_uram_store<0, 0, 0>((void*)0, buffer + buffer_size);
    __tx_fence();
    for(int i = 0; i < buffer_size; i++) {
        check(buffer[buffer_size + i], (char)i);
        buffer[buffer_size + i] = -1;
    }

    // 2D Load/Store
    __tx_tcsr_write<TCSR_DIM0>(buffer_size_2d0);
    __tx_tcsr_write<TCSR_STRIDE0>(buffer_size_2d1);
    __tx_tcsr_write<TCSR_DIM1>(buffer_size_2d1);
    __tx_tcsr_write<TCSR_STRIDE1>(1);
    printf("Plain 2D Load/Store(Transposed) begin\n");
    __tx_async_uram_load<0, 0, 0>(buffer, (void*)0);
    __tx_fence();
    __tx_async_uram_store<0, 0, 0>((void*)0, buffer + buffer_size);
    __tx_fence();
    for(int i = 0; i < buffer_size; i++) {
        check(buffer[buffer_size + i], (char)i);
        buffer[buffer_size + i] = -1;
    }


    printf("test_tx_load_store passed\n");
    return 0;
}