CXX := g++
CC  := gcc

CXXFLAGS := -g -Wall -std=c++17

BUILD_DIR := build

COMMON_SOURCES := common.cpp
TEST_SOURCES := $(wildcard test_*.cpp)

COMMON_OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(COMMON_SOURCES))
TEST_OBJECTS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(TEST_SOURCES))

TARGETS := $(patsubst %.cpp, $(BUILD_DIR)/%, $(TEST_SOURCES))


.PHONY: all clean

all: $(TARGETS)

$(TARGETS): %: %.o $(COMMON_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(COMMON_OBJ)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)