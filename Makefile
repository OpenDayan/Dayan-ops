CROSS_COMPILE=riscv64-unknown-elf-
CFLAGS=-g -Itestdata_gen -mcmodel=medany -std=gnu99 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g
CXXFLAGS=-Itestdata_gen -mcmodel=medany -std=c++17 -Ofast -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64g -Wa,-march=rv64g
LDFLAGS=-static -nostdlib

CXX        := $(CROSS_COMPILE)g++
CC         := $(CROSS_COMPILE)gcc
LD         := $(CROSS_COMPILE)ld
OBJDUMP    := $(CROSS_COMPILE)objdump

export CC CXX LD OBJDUMP CFLAGS CXXFLAGS LDFLAGS

modules=kernel common_fpga testdata_gen
link_script=-T common_fpga/test.ld

.PHONY: all $(modules) clean objdump testdata_gen common_fpga

.ONESHELL:
all: general_entry_fpga.packed.c
general_entry_fpga.elf: $(modules)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ common_fpga/*.o kernel/*.o -T common_fpga/test.ld
general_entry_fpga.packed.c: general_entry_fpga.elf
	python3 NPULinker.py $< $@

all: objdump

objdump:
	@find * \( -name '*.elf' \) -exec sh -c \
	'$(OBJDUMP) {} -aldS > {}.objdump && echo {}.objdump' ';'

$(modules):
	$(MAKE) --directory=$@

clean:
	for d in *; do
		if [ -f $$d/Makefile ]; then
			$(MAKE) --directory=$$d clean
		fi
	done
	rm -rf *.o *.elf *.objdump *.bin *.gen *.packed.c *~
	find . -name '*.objdump' -exec rm {} ';'

