ifeq ($(EMU),1)
CROSS_COMPILE=
CFLAGS=-g -std=gnu99
CXXFLAGS=-g -Iinlcude -std=c++11
else
CROSS_COMPILE=riscv64-unknown-elf-
CFLAGS=-mcmodel=medany -std=gnu99 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g -ffunction-sections
CXXFLAGS=-Iinlcude -mcmodel=medany -std=c++11 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g -ffunction-sections
LDFLAGS=-static -nostdlib -ffunction-sections -Wl,-gc-sections
endif

CXX        := $(CROSS_COMPILE)g++
CC         := $(CROSS_COMPILE)gcc
LD         := $(CROSS_COMPILE)ld
OBJDUMP    := $(CROSS_COMPILE)objdump

export CC CXX LD OBJDUMP CFLAGS CXXFLAGS LDFLAGS

tests       := matmul_test tcsr_xchg_test load_store_test
tests_elf   := $(tests:=.elf)
modules     := kernel

ifeq ($(EMU),1)
    EMU_DEFINE=-D__EMULATED_RVT_EXT__
else
    modules=kernel common
    link_script=-T common/test.ld
endif

CFLAGS+=$(EMU_DEFINE)
CXXFLAGS+=$(EMU_DEFINE)
objects     := $(addsuffix /*.o, $(modules))


.PHONY: all $(modules) clean objdump

.ONESHELL:
all: $(tests_elf) objdump

objdump: $(tests_elf)
	@find * \( -name '*.elf' \) -exec sh -c \
	'$(OBJDUMP) {} -aldS > {}.objdump && echo {}.objdump' ';'

$(modules):
	$(MAKE) --directory=$@

%.elf: testbench/%.c $(modules)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(objects) $(link_script)

clean:
	for d in *; do
		if [ -f $$d/Makefile ]; then
			$(MAKE) --directory=$$d clean
		fi
	done
	rm -rf *.o *.elf *.objdump *~
	find . -name '*.objdump' -exec rm {} ';'

