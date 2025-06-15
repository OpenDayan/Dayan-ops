ifeq ($(EMU),1)
CROSS_COMPILE=
CFLAGS=-Itestdata_gen -g -std=gnu99 -march=znver4
CXXFLAGS=-Itestdata_gen -g -std=c++17 -march=znver4
else
CROSS_COMPILE=riscv64-unknown-elf-
CFLAGS=-Itestdata_gen -mcmodel=medany -std=gnu99 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g
CXXFLAGS=-Itestdata_gen -mcmodel=medany -std=c++17 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g
LDFLAGS=-static -nostdlib
endif

CXX        := $(CROSS_COMPILE)g++
CC         := $(CROSS_COMPILE)gcc
LD         := $(CROSS_COMPILE)ld
OBJDUMP    := $(CROSS_COMPILE)objdump

export CC CXX LD OBJDUMP CFLAGS CXXFLAGS LDFLAGS

tests       := matmul_test matmul_at_test matmul_e4m3_bt_test tcsr_xchg_test load_store_test matmul_e4m3q_bt_test
tests_elf   := $(tests:=.elf)
modules     := kernel

ifeq ($(EMU),1)
    EMU_DEFINE=-D__EMULATED_RVT_EXT__
else
    EMU_DEFINE=-D__EMULATED_RVT_EXT__
    modules=kernel common
    link_script=-T common/test.ld
endif

CFLAGS+=$(EMU_DEFINE)
CXXFLAGS+=$(EMU_DEFINE)
objects     := $(addsuffix /*.o, $(modules))


.PHONY: all $(modules) clean objdump testdata_gen common_fpga

.ONESHELL:
ifneq ($(EMU),1)
all: general_entry_fpga.elf general_entry_fpga.packed.c
general_entry_fpga.elf: kernel common_fpga
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ common_fpga/*.o kernel/*.o -T common_fpga/test.ld
general_entry_fpga.packed.c: general_entry_fpga.elf
	python3 NPULinker.py $< $@
endif

all: $(tests_elf) objdump

objdump: $(tests_elf)
	@find * \( -name '*.elf' \) -exec sh -c \
	'$(OBJDUMP) {} -aldS > {}.objdump && echo {}.objdump' ';'

$(modules) testdata_gen common_fpga:
	$(MAKE) --directory=$@

%.elf: testbench/%.c $(modules) testdata_gen
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(objects) $(link_script)

clean:
	for d in *; do
		if [ -f $$d/Makefile ]; then
			$(MAKE) --directory=$$d clean
		fi
	done
	rm -rf *.o *.elf *.objdump *.bin *.gen *.packed.c *~
	find . -name '*.objdump' -exec rm {} ';'

