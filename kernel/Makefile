INCLUDES := -I../include/
target   := silu.o rms_norm.o elementwise.o get_rows.o matmul.o matmul_e4m3_q.o matmul_gemmini.o matmul_gemmini_e4m3_q.o instruction_testsuit.o \
            kernel_testsuit.o matmul_instruction_test.o elementwise_instruction_test.o convert_instruction_test.o lut_instruction_test.o \
						sfu_instruction_test.o rope.o softmax.o dup.o permute.o conv_instruction_test.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

.PHONY: all clean

all: $(target)

clean:
	rm -rf *.o