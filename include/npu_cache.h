#ifndef _NPU_CACHE_H_
#define _NPU_CACHE_H_

#define NPU_CACHE_LINE_SIZE 64
#define NPU_CACHE_LINE_COUNT 64UL
#define NPU_CACHE_WAY_COUNT   8UL

#ifdef __ASSEMBLER__

#define NPU_CBO_INVAL(_addr_reg) .insn i 0x0f, 2, x0, _addr_reg, 0x000
#define NPU_CBO_FLUSH(_addr_reg) .insn i 0x0f, 2, x0, _addr_reg, 0x002

#define NPU_CACHE_INVAL_RANGE(_start, _end, _line_reg, _end_reg)                                    \
    la _line_reg, _start;                                                                          \
    la _end_reg, _end;                                                                             \
    addi _end_reg, _end_reg, NPU_CACHE_LINE_SIZE - 1;                                              \
    andi _line_reg, _line_reg, -NPU_CACHE_LINE_SIZE;                                               \
    andi _end_reg, _end_reg, -NPU_CACHE_LINE_SIZE;                                                 \
    fence rw, rw;                                                                                  \
    1 : bgeu _line_reg, _end_reg, 2f;                                                              \
    NPU_CBO_INVAL(_line_reg);                                                                      \
    addi _line_reg, _line_reg, NPU_CACHE_LINE_SIZE;                                                \
    j 1b;                                                                                          \
    2 : fence rw, rw

#define NPU_CACHE_FLUSH_RANGE(_start, _end, _line_reg, _end_reg)                                   \
    la _line_reg, _start;                                                                          \
    la _end_reg, _end;                                                                             \
    addi _end_reg, _end_reg, NPU_CACHE_LINE_SIZE - 1;                                              \
    andi _line_reg, _line_reg, -NPU_CACHE_LINE_SIZE;                                               \
    andi _end_reg, _end_reg, -NPU_CACHE_LINE_SIZE;                                                 \
    fence rw, rw;                                                                                  \
    1 : bgeu _line_reg, _end_reg, 2f;                                                              \
    NPU_CBO_FLUSH(_line_reg);                                                                      \
    addi _line_reg, _line_reg, NPU_CACHE_LINE_SIZE;                                                \
    j 1b;                                                                                          \
    2 : fence rw, rw

#else

#include <stddef.h>
#include <stdint.h>

static inline void cache_inval_line(const void *addr) {
    uintptr_t line = (uintptr_t)addr & ~(NPU_CACHE_LINE_SIZE - 1);
    __asm__ __volatile__(".insn i 0x0f, 2, x0, %0, 0x000" : : "r"(line) : "memory"); // cbo.inval
}

static inline void cache_inval_range(const void *addr, size_t size) {
    if (size == 0) {
        return;
    }

    uintptr_t start = (uintptr_t)addr & ~(NPU_CACHE_LINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + NPU_CACHE_LINE_SIZE - 1) & ~(NPU_CACHE_LINE_SIZE - 1);

    __asm__ __volatile__("fence rw, rw" : : : "memory");
    for (uintptr_t line = start; line < end; line += NPU_CACHE_LINE_SIZE) {
        cache_inval_line((const void *)line);
    }
    __asm__ __volatile__("fence rw, rw" : : : "memory");
}

static inline void cache_flush_line(const void *addr) {
    uintptr_t line = (uintptr_t)addr & ~(NPU_CACHE_LINE_SIZE - 1);
    __asm__ __volatile__(".insn i 0x0f, 2, x0, %0, 0x002" : : "r"(line) : "memory"); // cbo.flush
}

static inline void cache_flush_range(const void *addr, size_t size) {
    if (size == 0) {
        return;
    }

    uintptr_t start = (uintptr_t)addr & ~(NPU_CACHE_LINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + NPU_CACHE_LINE_SIZE - 1) & ~(NPU_CACHE_LINE_SIZE - 1);

    __asm__ __volatile__("fence rw, rw" : : : "memory");
    for (uintptr_t line = start; line < end; line += NPU_CACHE_LINE_SIZE) {
        cache_flush_line((const void *)line);
    }
    __asm__ __volatile__("fence rw, rw" : : : "memory");
}

#endif // __ASSEMBLER__

#endif // _NPU_CACHE_H_
