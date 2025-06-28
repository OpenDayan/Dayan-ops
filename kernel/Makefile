INCLUDES := -I../include/
target   := matmul.o matmul_e4m3_q.o matmul_gemmini.o matmul_gemmini_e4m3_q.o instruction_testsuit.o emu_support.o \
            kernel_testsuit.o matmul_instruction_test.o elementwise_instruction_test.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

.PHONY: all clean

all: $(target)

clean:
	rm -rf *.o