INCLUDES := -I../include/
target   := test_misc_kernels.o test_task_kernels.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

.PHONY: all clean

all: $(target)

clean:
	rm -rf *.o