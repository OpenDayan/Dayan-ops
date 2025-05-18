CFLAGS = -Ofast -std=c11 -march=rv64g
CXXFLAGS = -Ofast -std=c++11 -march=rv64g

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
ASMS = $(SRCS:.cpp=.S)
DEPS = $(SRCS:.cpp=.d)

CXX = $(CROSS_COMPILE)g++ $(CXXFLAGS)
OBJDUMP = $(CROSS_COMPILE)objdump

all: $(ASMS)

%.s:%.cpp
	$(CXX) -S $< -o $@
%.o:%.cpp
	$(CXX) -c $<
%.S:%.o
	$(OBJDUMP) -S $< > $@
%.d: %.cpp
	rm -f $@; \
	$(CXX) -MM $< >$@.tmp; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@; \
	rm -f $@.tmp

clean:
	rm -f *.o *.S *.s *.d

include $(DEPS)
