#!/bin/bash

# terminate.sh - 终止 pre-run.sh 及其执行的进程

# 获取脚本名称（用于查找进程）
SCRIPT_NAME=$(cat .vscode.prerun.name)
TERMINATE_SCRIPT_NAME="vscode-post-run.sh"

# 函数：显示用法
usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -v, --verbose    显示详细信息"
    echo "  -f, --force      强制终止进程（使用 SIGKILL）"
    echo "  -h, --help       显示此帮助信息"
}

# 函数：终止进程及其子进程
terminate_process_tree() {
    local pid=$1
    local signal=${2:-SIGTERM}
    local verbose=$3
    
    if [ -z "$pid" ] || [ "$pid" -le 1 ]; then
        return
    fi
    
    # 获取所有子进程
    local child_pids=$(ps -o pid --no-headers --ppid "$pid" 2>/dev/null)
    
    # 先终止子进程
    for child_pid in $child_pids; do
        terminate_process_tree "$child_pid" "$signal" "$verbose"
    done
    
    # 然后终止父进程
    if ps -p "$pid" > /dev/null 2>&1; then
        if [ "$verbose" = "true" ]; then
            echo "发送 $signal 到进程 $pid ($(ps -p "$pid" -o comm= 2>/dev/null))"
        fi
        kill -"$signal" "$pid" 2>/dev/null
        
        # 如果是 SIGTERM，等待进程退出
        if [ "$signal" = "SIGTERM" ]; then
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 5 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # 如果进程仍然存在，使用 SIGKILL
            if ps -p "$pid" > /dev/null 2>&1; then
                if [ "$verbose" = "true" ]; then
                    echo "进程 $pid 未响应 SIGTERM，发送 SIGKILL"
                fi
                kill -SIGKILL "$pid" 2>/dev/null
            fi
        fi
    fi
}

# 解析命令行参数
VERBOSE=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            usage
            exit 1
            ;;
    esac
done

if [ "$VERBOSE" = "true" ]; then
    echo "正在查找并终止 pre-run.sh 相关进程..."
fi

# 查找 pre-run.sh 进程
pids=$(ps aux | grep "$SCRIPT_NAME" | grep -v grep | grep -v "$TERMINATE_SCRIPT_NAME" | awk '{print $2}')

if [ -z "$pids" ]; then
    if [ "$VERBOSE" = "true" ]; then
        echo "未找到运行的 pre-run.sh 进程"
    fi
    exit 0
fi

# 确定要使用的信号
if [ "$FORCE" = "true" ]; then
    SIGNAL="SIGKILL"
else
    SIGNAL="SIGTERM"
fi

# 终止找到的每个进程及其子进程
for pid in $pids; do
    if [ "$VERBOSE" = "true" ]; then
        echo "找到进程 $pid，开始终止进程树..."
    fi
    
    terminate_process_tree "$pid" "$SIGNAL" "$VERBOSE"
done

if [ "$VERBOSE" = "true" ]; then
    echo "所有相关进程已终止"
fi

# 额外清理：终止可能在 make 过程中创建的进程
# 查找在 test_instructions 目录中运行的 make 进程
make_pids=$(ps aux | grep "make.*test_instructions" | grep -v grep | awk '{print $2}')
for pid in $make_pids; do
    if [ "$VERBOSE" = "true" ]; then
        echo "终止残留的 make 进程: $pid"
    fi
    kill -SIGTERM "$pid" 2>/dev/null
done

exit 0
