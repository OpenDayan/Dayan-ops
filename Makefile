ifeq ($(EMU),1)
CROSS_COMPILE=
CFLAGS=-Itestdata_gen -g -std=gnu99
CXXFLAGS=-Itestdata_gen -g -std=c++17
else
CROSS_COMPILE=riscv64-unknown-elf-
CFLAGS=-Itestdata_gen -mcmodel=medany -std=gnu99 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g -ffunction-sections
CXXFLAGS=-Itestdata_gen -mcmodel=medany -std=c++11 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g -ffunction-sections
LDFLAGS=-static -nostdlib -ffunction-sections -Wl,-gc-sections
endif

CXX        := $(CROSS_COMPILE)g++
CC         := $(CROSS_COMPILE)gcc
LD         := $(CROSS_COMPILE)ld
OBJDUMP    := $(CROSS_COMPILE)objdump

export CC CXX LD OBJDUMP CFLAGS CXXFLAGS LDFLAGS

tests       := matmul_test matmul_at_test matmul_e4m3_at_test tcsr_xchg_test load_store_test
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


.PHONY: all $(modules) clean objdump testdata_gen

.ONESHELL:
all: $(tests_elf) objdump

objdump: $(tests_elf)
	@find * \( -name '*.elf' \) -exec sh -c \
	'$(OBJDUMP) {} -aldS > {}.objdump && echo {}.objdump' ';'

$(modules) testdata_gen:
	$(MAKE) --directory=$@

%.elf: testbench/%.c $(modules) testdata_gen
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(objects) $(link_script)

clean:
	for d in *; do
		if [ -f $$d/Makefile ]; then
			$(MAKE) --directory=$$d clean
		fi
	done
	rm -rf *.o *.elf *.objdump *~
	find . -name '*.objdump' -exec rm {} ';'

