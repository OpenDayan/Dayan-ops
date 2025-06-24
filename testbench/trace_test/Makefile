CXX := g++
CC  := gcc

CXXFLAGS := -g -Wall -std=c++17

BIN_DIR := bin
COMMON_SOURCES := common.cpp
TEST_SOURCES := $(wildcard ./test_*.cpp)
TEST_OBJECTS := $(patsubst ./%.cpp, ./%.o, $(TEST_SOURCES))
TEST_TARGETS := $(patsubst ./%.cpp, $(BIN_DIR)/%, $(TEST_SOURCES))

.PHONY: all

all: $(TEST_TARGETS)

$(BIN_DIR)/%: %.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(COMMON_SOURCES)

clean:
	rm -rf $(BIN_DIR)