INCLUDES := -I../include/
target   := matmul.o instruction_testsuit.o emu_support.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

.PHONY: all clean

all: $(target)

clean:
	rm -rf *.o