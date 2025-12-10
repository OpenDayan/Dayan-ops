#pragma once

enum MAT_TCSR {
    MAT_M = 0,
    MAT_N = 1,
    MAT_K = 2,
    MAT_ASTRIDE = 3,
    MAT_BSTRIDE = 4,
    MAT_CSTRIDE = 5,
    MAT_ASWIZZLEMBS = 6,
    MAT_BSWIZZLEMBS = 7,
    MAT_CSWIZZLEMBS = 8,
    MAT_PERSISTENTMODE = 9,
    MAT_DATATYPE = 10,
    MAT_CHECKSUM = 31
};

#define MAT_GROUPSZ (32)
static inline long mat_batch_swizzle(uint8_t M, uint8_t B, uint8_t S) {
    return ((uint64_t)M << 16) | ((uint64_t)B << 8) | ((uint64_t)S);
}
