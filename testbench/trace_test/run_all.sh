#!/bin/bash

make clean && make -j8

./build/test_silu_f32 20

./build/test_dup_f32_f16 30

./build/test_add_f32 30

./build/test_rms_norm_f32 20

./build/test_mul_f32 30

./build/test_rope_f32_i32 20

./build/test_softmax_f32 30

./build/test_get_rows_f32 6

./build/test_matmul_f32 30
