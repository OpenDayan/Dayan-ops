#include "npu_kernel.h"
#include <stddef.h>
#include <stdint.h>

__device__ void *kernel_memset(void *dest, long byte, size_t len) {
    if ((((uintptr_t)dest | len) & (sizeof(uintptr_t) - 1)) == 0) {
        uintptr_t word = byte & 0xFF;
        word |= word << 8;
        word |= word << 16;
        word |= word << 16 << 16;

        uintptr_t *d = dest;
        while (d < (uintptr_t *)(dest + len))
            *d++ = word;
    } else {
        char *d = dest;
        while (d < (char *)(dest + len))
            *d++ = byte;
    }
    return dest;
}

__device__ void __tx_init() {
    // Do nothing
}
