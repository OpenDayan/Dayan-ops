INCLUDES := -I../include/
target   := matmul.o matmul_e4m3_q.o instruction_testsuit.o emu_support.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

.PHONY: all clean

all: $(target)

clean:
	rm -rf *.o