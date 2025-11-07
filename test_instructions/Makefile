CXX := g++
CC  := gcc

BUILD_DIR := build
RUNTIME_DIR := ../../npu-runtime
KERNEL_PACK := ../general_entry_fpga.packed.c

COMMON_SOURCES :=
TEST_SOURCES := $(wildcard test_*.cpp)

COMMON_OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(COMMON_SOURCES)) $(BUILD_DIR)/general_entry_fpga.packed.o $(BUILD_DIR)/NPUDriver.o \
                                                                         $(BUILD_DIR)/NPURuntime.o $(BUILD_DIR)/NPUMemoryManagement.o
TEST_OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(TEST_SOURCES))

TARGETS := $(patsubst %.cpp, $(BUILD_DIR)/%, $(TEST_SOURCES))

CFLAGS := -g -Wall -O3 -std=gnu99 -I$(RUNTIME_DIR)/include -I../testdata_gen -I../include
CXXFLAGS := -g -Wall -O3 -std=c++17 -I$(RUNTIME_DIR)/include -I../testdata_gen -I../include

.PHONY: all clean $(KERNEL_PACK)

all: $(TARGETS)

$(TARGETS): %: %.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(COMMON_OBJ)

$(BUILD_DIR)/NPUDriver.o: $(RUNTIME_DIR)/src/NPUDriver.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/NPURuntime.o: $(RUNTIME_DIR)/src/NPURuntime.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/NPUMemoryManagement.o: $(RUNTIME_DIR)/src/NPUMemoryManagement.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/general_entry_fpga.packed.o: $(KERNEL_PACK)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(KERNEL_PACK):
	$(MAKE) --directory=../

clean:
	rm -rf $(BUILD_DIR)

qemu:
	umount /tmp/virtual_npu; /workspace/npu-qemu/build/qemu-system-riscv64 -monitor null -bios /workspace/npu-firmware/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096 -S -s

.ONESHELL:
gdb:
	export QEMU="/workspace/npu-qemu/build/qemu-system-riscv64"
	export QEMU_FLAGS="-bios /workspace/npu-firmware/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096"
	setsid ./run_bg.sh $$$$ &
	exec gdb-multiarch -q /workspace/npu-kernels/general_entry_fpga.elf -ex "target remote localhost:1234" -ex "b *0x80000000"
