# Future-NPU 裸机算子测试

## 构建 RISC-V 指令集下的算子测试目标

直接使用 `make` 即可针对 rv64g + rvt 指令集构建所有 testbench 目录下的测试目标，并在当前文件夹下生成测试 elf 和反汇编 objdump 文件。

目前裸机环境针对 chipyard 仿真，在 `chipyard/sims/verilator` 目录下使用命令 `make -j CONFIG=GemminiRocketConfig  run-binary LOADMEM=1 BINARY=${elf_file}`，将 `elf_file` 修改为期望使用的测试 elf 的绝对路径即可。

参考指令：`make clean && make`

## 构建本地平台（x64）下的算子测试目标

为了方便及加速算子调试，裸机算子测试支持在本地 x64 Linux 环境下编译运行。只需要为 `make` 添加参数 `EMU=1` 即可为 x64 构建测试 elf。

**注意：构建不同体系结构下目标前，建议先执行 `make clean`**

参考指令：`make clean && EMU=1 make`
