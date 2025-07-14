CXX := g++
CC  := gcc

BUILD_DIR := build
RUNTIME_DIR := ../../npu-runtime
KERNEL_PACK := ../general_entry_fpga.packed.c

COMMON_SOURCES := common.cpp
TEST_SOURCES := $(wildcard test_*.cpp)

COMMON_OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(COMMON_SOURCES)) $(BUILD_DIR)/general_entry_fpga.packed.o $(BUILD_DIR)/NPUDriver.o $(BUILD_DIR)/NPURuntime.o
TEST_OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(TEST_SOURCES))

TARGETS := $(patsubst %.cpp, $(BUILD_DIR)/%, $(TEST_SOURCES))

CFLAGS := -g -Wall -std=c11 -I$(RUNTIME_DIR)/include
CXXFLAGS := -g -Wall -std=c++17 -I$(RUNTIME_DIR)/include

.PHONY: all clean $(KERNEL_PACK)

all: $(TARGETS)

$(TARGETS): %: %.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(COMMON_OBJ)

$(BUILD_DIR)/NPUDriver.o: $(RUNTIME_DIR)/src/NPUDriver.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/NPURuntime.o: $(RUNTIME_DIR)/src/NPURuntime.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/general_entry_fpga.packed.o: $(KERNEL_PACK)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(KERNEL_PACK):
	$(MAKE) --directory=../

clean:
	rm -rf $(BUILD_DIR)