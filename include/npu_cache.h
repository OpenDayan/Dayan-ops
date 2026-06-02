#ifndef _NPU_CACHE_H_
#define _NPU_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#define NPU_CACHE_LINE_SIZE 64UL

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

#endif // _NPU_CACHE_H_
