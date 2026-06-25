#!/bin/bash

# --------------- 遍历所有算子的所有 trace 进行测试 -----------------

# Usage: ./run_full.sh
# ./build/test_add ./trace/decode10/add  0

# ===== CONFIG: =====
# prefill seq-len
prefill_num=6
# decode seq-len
decode_num=1
# trace根目录
trace_dir=/public/trace0104

# 算子列表 （add、dup、mul、silu、rms_norm、matmul、softmax、rope、get_rows、permute）
op_list=(add dup mul silu rms_norm matmul softmax rope get_rows permute)

# 每一个 Decode Step算子调用数量 (Qwen2为例：)
declare -A op_num_per_step=(
    [add]=140
    [dup]=84
    [mul]=85
    [silu]=28
    [rms_norm]=57
    [matmul]=253
    [softmax]=28
    [rope]=56
    [get_rows]=3
    [permute]=56
)


# ===== RUN: =====
run_test() {
    local op_name=$1
    local trace_sub_dir=$2
    local num=$3
    local test_bin=./build/test_${op_name}
    for ((i=0; i<num; i++)); do
        $test_bin $trace_sub_dir $i > /dev/null;
        ret=$?
        if [ $ret -eq 0 ]; then
            if (( i % 20 == 0 )); then
                echo "[INFO] Testing $op_name trace-$i"
            fi
        else
            echo "[INFO] Test $op_name failed at trace-$i"
            exit 1
        fi
    done

    return $?
}

for op_name in "${op_list[@]}"; do
    num=${op_num_per_step[$op_name]}
    echo " ----------------------- Test OP: $op_name ($num/step) ------------------------ "

    # echo ">>>>>>>>>>>> Start test with Prefill Stage trace"
    # run_test "$op_name" "$trace_dir/prefill0-$(($prefill_num-1))/$op_name" $num
    # echo ">>>>>>>>>>>> End test with prefill stage trace"

    for ((j=prefill_num; j<$(($prefill_num + $decode_num)); j++)); do
        echo ">>>>>>>>>>>> Start test with Decode-$j trace"
        run_test "$op_name" "$trace_dir/decode$j/$op_name" $num
        echo ">>>>>>>>>>>> End test with Token-$j trace"
    done
done

echo "[INFO] All operators passed!"
