#!/bin/bash

make -j

tar -xf example_trace.tar

# stage=decode6
stage=prefill0-5

./build/test_add example_trace/${stage}/add/ 0
./build/test_add example_trace/${stage}/add/ 1

./build/test_dup example_trace/${stage}/dup/ 0
./build/test_dup example_trace/${stage}/dup/ 1
./build/test_dup example_trace/${stage}/dup/ 2

./build/test_get_rows example_trace/${stage}/get_rows/ 0

./build/test_matmul example_trace/${stage}/matmul/ 0
./build/test_matmul example_trace/${stage}/matmul/ 1

./build/test_mul example_trace/${stage}/mul/ 0
./build/test_mul example_trace/${stage}/mul/ 1

./build/test_permute example_trace/${stage}/permute/ 0
./build/test_permute example_trace/${stage}/permute/ 1

./build/test_rms_norm example_trace/${stage}/rms_norm/ 0
./build/test_rms_norm example_trace/${stage}/rms_norm/ 1

./build/test_rope example_trace/${stage}/rope/ 0
./build/test_rope example_trace/${stage}/rope/ 1

./build/test_silu example_trace/${stage}/silu/ 0
./build/test_silu example_trace/${stage}/silu/ 1

./build/test_softmax example_trace/${stage}/softmax/ 0
./build/test_softmax example_trace/${stage}/softmax/ 1