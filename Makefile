# 根目录 Makefile
.PHONY: all clean distclean $(SUB_DIRS) objdump pack

# 项目配置
PROJECT_NAME := kernels_pack
SUB_DIRS := kernel libs
BUILD_DIR := build
OUTPUT_DIR := bin

# 工具链配置
CROSS_COMPILE ?= riscv64-unknown-elf-
CC := $(CROSS_COMPILE)gcc
CXX := $(CROSS_COMPILE)g++
LD := $(CROSS_COMPILE)ld
OBJDUMP := $(CROSS_COMPILE)objdump
PYTHON := python3

# 编译选项
CFLAGS := -g -mcmodel=medany -std=gnu99 -O3 -ffast-math \
          -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
          -fno-pic -march=rv64g -Wa,-march=rv64g

CXXFLAGS := -g -mcmodel=medany -std=c++17 -O3 -ffast-math \
            -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
            -fno-pic -march=rv64g -Wa,-march=rv64g

CPPFLAGS := -I./include
LDFLAGS := -static -nostdlib

# 导出变量到子 Makefile
export CC CXX LD OBJDUMP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS

# 链接脚本
LINK_SCRIPT := libs/test.ld

# 输出文件
ELF_FILE := $(OUTPUT_DIR)/$(PROJECT_NAME).elf
PACKED_FILE := $(OUTPUT_DIR)/$(PROJECT_NAME).packed.c

# 默认目标
all: $(OUTPUT_DIR) $(BUILD_DIR) $(SUB_DIRS) $(ELF_FILE) pack objdump

$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(SUB_DIRS): $(BUILD_DIR)
	@echo "Building $@..."
	$(MAKE) -C $@ BUILD_DIR=../$(BUILD_DIR)

$(ELF_FILE): $(SUB_DIRS)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ \
		$(BUILD_DIR)/*.o \
		-T $(LINK_SCRIPT)

# 生成 .packed.c 文件
pack: $(PACKED_FILE)
$(PACKED_FILE): $(ELF_FILE)
	@echo "Generating packed C file..."
	$(PYTHON) NPULinker.py $< $@

# 生成 objdump 文件
objdump: $(ELF_FILE)
	@echo "Generating objdump..."
	$(OBJDUMP) $(ELF_FILE) -aldS > $(ELF_FILE).objdump

# 清理目标
clean:
	@for dir in $(SUB_DIRS); do \
		echo "Cleaning $$dir..."; \
		$(MAKE) -C $$dir clean; \
	done
	@rm -rf $(BUILD_DIR) $(OUTPUT_DIR)
	@find . -name '*.objdump' -exec rm -f {} +
	@echo "Clean completed"

distclean: clean
	@echo "Distclean completed"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  clean     - Remove object files and temporary files"
	@echo "  distclean - Remove all generated files"
	@echo "  objdump   - Dump ELF to assembly"
	@echo "  pack      - Generate packed C file from ELF"
	@echo "  help      - Show this help message"
