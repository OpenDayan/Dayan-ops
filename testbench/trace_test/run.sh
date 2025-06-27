#!/bin/bash

# --------------- Example1: single run -----------------

# Usage: ./test <TRACE SUB DIR> <TRACE ID>
# ./build/test_add_f32 ./trace/decode10/add  0

# --------------- Example2: batch run ----------------------

# ===== CONFIG: =====
# prefill seq-len
prefill_num=6
# decode seq-len
decode_num=1
# trace根目录
trace_dir=trace
# 算子名称 （add、dup、mul、silu、rms_norm、matmul、softmax、rope、get_rows）
op_name=softmax
# 每一个 Decode Step算子调用数量 (Qwen2为例：)
# add:      140
# dup:      84
# mul:      85
# silu:     28
# rms_norm: 57
# matmul:   253
# softmax:  28
# rope:     56
# get_rows: 3
op_num_per_step=28
# 编译后的测试程序 (与 op_name 算子类型一致)
test_bin=./build/test_softmax_f32


# ===== RUN: =====
run_test() {
    local test_bin=$1
    local trace_sub_dir=$2
    local trace_num=$3
    for ((i=0; i<$op_num_per_step; i++)); do
        $test_bin $trace_sub_dir $i > /dev/null;
        ret=$?
        if [ $ret -eq 0 ]; then
            if (( i % 20 == 0 )); then
                echo "[INFO] Testing trace-$i"
            fi
        else
            echo "[INFO] Test $op_name failed at trace-$i"
            exit 1
        fi
    done


    return $?
}

echo " ----------------------- Test OP: $op_name ------------------------ "

echo ">>>>>>>>>>>> Start test with Prefill Stage trace"
run_test "$test_bin" "$trace_dir/prefill0-$(($prefill_num-1))/$op_name" $op_num_per_step
echo ">>>>>>>>>>>> End test with prefill stage trace"

for ((j=prefill_num; j<$(($prefill_num + $decode_num)); j++)); do
    echo ">>>>>>>>>>>> Start test with Decode-$j trace"
    run_test "$test_bin" "$trace_dir/decode$j/$op_name" $op_num_per_step
    echo ">>>>>>>>>>>> End test with Token-$j trace"
done