
CFLAGS        +=    -mcmodel=medany \
                    -std=gnu99 \
                    -Ofast \
                    -ffast-math \
                    -fno-common \
                    -fno-builtin-printf \
                    -fno-tree-loop-distribute-patterns \
                    -march=rv64g -Wa,-march=rv64g \

CXXFLAGS      +=    -Iinlcude \
					-mcmodel=medany \
                    -std=c++11 \
                    -Ofast \
                    -ffast-math \
                    -fno-common \
                    -fno-builtin-printf \
                    -fno-tree-loop-distribute-patterns \
                    -march=rv64g -Wa,-march=rv64g \

LDFLAGS += -static -nostdlib

CROSS_COMPILE ?= riscv64-unknown-elf-

CXX        := $(CROSS_COMPILE)g++
CC         := $(CROSS_COMPILE)gcc
LD         := $(CROSS_COMPILE)ld
OBJDUMP    := $(CROSS_COMPILE)objdump

export CC CXX LD OBJDUMP CFLAGS CXXFLAGS LDFLAGS

tests       := matmul_test
tests_elf   := $(tests:=.elf)
modules     := common kernel
objects     := $(addsuffix /*.o, $(modules))
link_script := common/test.ld

.PHONY: all $(modules) testbench clean objdump

.ONESHELL:
all: $(tests_elf) objdump

objdump: $(tests_elf)
	@find * \( -name '*.elf' \) -exec sh -c \
	'$(OBJDUMP) {} -aldS > {}.objdump && echo {}.objdump' ';'

$(modules):
	$(MAKE) --directory=$@

%.elf: testbench/%.c $(modules)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(objects) -T $(link_script)

clean:
	for d in *; do
		if [ -f $$d/Makefile ]; then
			$(MAKE) --directory=$$d clean
		fi
	done
	rm -rf *.o *~
	find . -name '*.objdump' -exec rm {} ';'

