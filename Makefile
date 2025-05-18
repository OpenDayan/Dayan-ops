CFLAGS = -Ofast -std=c11 -march=rv64g
CXXFLAGS = -Ofast -std=c++11 -march=rv64g

CXX = $(CROSS_COMPILE)g++ $(CXXFLAGS)
CC = $(CROSS_COMPILE)gcc $(CFLAGS)
OBJDUMP = $(CROSS_COMPILE)objdump

all: npu_instruction.o npu_instruction.s npu_instruction.S

%.s:%.cpp
	$(CXX) -S $< -o $@
%.s:%.c
	$(CC) -S $< -o $@
%.o:%.cpp
	$(CXX) -c $<
%.o:%.c
	$(CC) -c $<
%.S:%.o
	$(OBJDUMP) -S $< > $@

clean:
	rm -f *.o *.S *.s
