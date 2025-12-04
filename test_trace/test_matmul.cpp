#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <x86intrin.h>

#include <NPURuntime.h>

#include "common.hpp"

#define QK 256

struct block_e4m3_q {
    float scale;
    uint8_t data[QK];
};

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                                                : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

float bf16_to_fp32(uint16_t h) {
    union {
        float value;
        uint32_t bits;
    } fp32;
    fp32.bits = (static_cast<uint32_t>(h) << 16);
    return fp32.value;
}

uint16_t fp32_to_fp16(float f) {
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

uint16_t fp32_to_bf16(float f) {
    uint16_t res;
    union {
        float value;
        uint32_t bits;
    } fp32;
    fp32.value = f;
    if ((fp32.bits & 0x7fffffff) > 0x7f800000) { /* nan */
        res = (fp32.bits >> 16) | 64;            /* force to quiet */
        return res;
    }
    res = (fp32.bits + (0x7fff + ((fp32.bits >> 16) & 1))) >> 16;
    return res;
}

uint16_t fp32_to_bf16_round(float f) {

    uint32_t x = 0;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = x & 0x80000000;
    int32_t exponent = (x >> 23) & 0xFF;
    // uint32_t mantissa = x & 0x007FFFFF;

    if (exponent == 0xFF) {
        return (x >> 16) | 0x007F;
    }

    if (exponent == 0) {
        return (x >> 16) & 0x8000;
    }

    uint32_t truncated = x >> 16;
    uint32_t round_bit = (x >> 15) & 1;
    uint32_t sticky_bits = x & 0x7FFF;
    // Round
    if (round_bit && (sticky_bits || (truncated & 1)))
        truncated += 1;
    // // Ceil
    // if (!sign && (round_bit || sticky_bits))
    //     truncated += 1;
    // // Floor
    // if (sign && (round_bit || sticky_bits))
    //     truncated += 1;

    if ((truncated & 0x7F80) == 0x7F80) {
        truncated = (truncated & 0x8000) | 0x7F80;
    }

    return (uint16_t)truncated;
}

union fp32_int32 {
    float f;
    uint32_t bits;
};

template <int N> constexpr int exp_i2() {
    return 1 << N;
}

template <int N> constexpr float exp_f2() {
    if constexpr (N > 0)
        return exp_f2<N - 1>() * 2;
    if constexpr (N < 0)
        return exp_f2<N + 1>() / 2;
    if constexpr (N == 0)
        return 1.;
}

template <int _E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<_E>;
    static constexpr int E = _E;
    static constexpr int M = (7 - _E);
    static constexpr int E_BIAS = exp_i2<E - 1>() - 1;
    static constexpr float MAX = (2 - exp_f2<-M + 1>()) * exp_f2<exp_i2<E - 1>()>();
    static constexpr float MIN = exp_f2<-M>() * exp_f2<2 - exp_i2<E - 1>()>();
};

template <int E> struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template <int E> inline FP8<E> float_to_fp8(float value) {
    FP8<E> out;
    fp32_int32 in = {value};
    // the sign
    out.bits = (in.bits >> 24) & 0x80;
    // value without sign
    in.bits &= 0x7fffffff;
    // GGML_ASSERT(in.bits < 0x7f800000); // +/- infinity or NAN
    if (in.f >= FP8<E>::MAX) {
        out.bits |= 0x7E;
    } else if (in.f < FP8<E>::MIN) { // => 0.
                                     // OK: S.0000000
    } else {
        in.f *= exp_f2<FP8<E>::E_BIAS - 127>();
        // - trunc
        // uint32_t eps = 0;
        // - rounding half away from zero
        // uint32_t eps = 0x400000>>FP8<E>::M;
        // - rounding half toward zero
        // uint32_t eps = 0x3fffff>>FP8<E>::M;
        // - rounding to nearest even
        uint32_t eps = (0x3fffff >> FP8<E>::M) + ((in.bits >> (23 - FP8<E>::M)) & 0x1);
        // shift mantissa.
        in.bits += eps;
        out.bits |= (in.bits >> (23 - FP8<E>::M)) & 0x7F;
    }
    return out;
}

template <int E> static inline void float_to_e4m3q(const float *x, bloc_fp8<E> *y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q = 0; q < qk_size; ++q) {
        float m = 0;
        for (int64_t i = 0; i < QK; i++) {
            m = std::max(std::abs(x[q * QK + i]), m);
        }
        const float D = FP8<E>::MAX / m;
        y[q].d = m / FP8<E>::MAX;
        for (int64_t i = 0; i < QK; i++) {
            y[q].qs[i] = float_to_fp8<E>(x[q * QK + i] * D);
        }
    }
}

float e4m3q_to_float(uint8_t h) {
    fp32_int32 out = {0};
    out.bits = h & 0x80;
    out.bits <<= 24;
    uint32_t h_bits = h & 0x7F;
    h_bits <<= (23 - 3); // 23 - FP8::M (M==3 in e4m3q)
    out.bits |= h_bits;
    // 2^(127 - FP8::E_BIAS)
    out.f *= std::pow(2, 127 - 7); // E==4 in e4m3q, so E_BIAS = 2^(E-1) - 1 = 7
    return out.f;
}

// (2 - std::exp2(-M + 1)) * std::exp2(std::exp2(E - 1)) = (2 - 2**(-2)) *(2**(2**(3))) = (2 -
// 2**(-2)) * 2**(8)
static const float e4m3q_max = (2 - std::exp2(-2)) * (std::exp2(8));

// (std::exp2(-M)) * (std::exp2(2 - std::exp2(E - 1))) = (2**(-3)) * (2**(2 - 2**3)) = 2**(-9)
static const float e4m3q_min = std::exp2(-3) * std::exp2(-6);
uint8_t float_to_e4m3q(float h) {
    uint8_t out;
    fp32_int32 in = {h};
    out = (in.bits >> 24) & 0x80;
    in.bits &= 0x7fffffff;
    if (in.f >= e4m3q_max) {
        out |= 0x7E;
    } else if (in.f < e4m3q_min) { // => 0.
    } else {
        in.f *= std::exp2(7 - 127); // E==4 in e4m3q, so E_BIAS = 2^(E-1) - 1 = 7
        uint32_t eps = (0x3fffff >> 3) + ((in.bits >> (23 - 3)) & 0x1);
        in.bits += eps;
        out |= (in.bits >> (23 - 3)) & 0x7F;
    }
    return out;
}

void block_float_to_e4m3q(float *b_f32, block_e4m3_q *b_fp8) {
    float m = 0;
    for (int i = 0; i < QK; ++i) {
        m = std::max(std::abs(b_f32[i]), m);
    }

    const float D = e4m3q_max / m;

    // set dequantize scale
    b_fp8->scale = m / e4m3q_max;

    for (int i = 0; i < QK; ++i) {
        b_fp8->data[i] = float_to_e4m3q(b_f32[i] * D);
    }
}

void matmul_bf16_f32_f32(std::vector<uint16_t> &a, std::vector<float> &b, std::vector<float> &res,
                         TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    int bs = psrc1.ne[3];
    int h = psrc1.ne[2];
    int h0 = psrc0.ne[2];

    // dim(0) and dim(1) only broadcast from src0 to src1
    int rbs = bs / psrc0.ne[3];
    int rj = h / psrc0.ne[2];
    assert(bs >= psrc0.ne[3]);
    assert(h >= psrc0.ne[2]);

    // src0 shape: (bs0, h0, n, k)
    // src1 shape: (bs, h, m, k)
    // dst shape: (bs, h, m, n)
    int n = psrc0.ne[1];
    int m = psrc1.ne[1];
    int k = psrc1.ne[0];
    assert(psrc1.ne[0] == psrc0.ne[0]);

    for (int i = 0; i < bs; ++i)
        for (int j = 0; j < h; ++j) {
            // broadcast in dim(0)
            int i0 = i / rbs;
            // broadcase in dim(1)
            int j0 = j / rj;

            // b[i][j][:][:] * a[i0][j0][:][:]^T
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    float val = 0.0;
                    // res[i][j][p][l] = b[i][j][p][:] * a[i0][j0][l][:]^T
                    for (int x = 0; x < k; ++x) {
                        uint16_t ea = a[(i0 * psrc0.nb[3] + j0 * psrc0.nb[2] + l * psrc0.nb[1] +
                                         x * psrc0.nb[0]) /
                                        sizeof(uint16_t)];
                        float eb = b[(i * psrc1.nb[3] + j * psrc1.nb[2] + p * psrc1.nb[1] +
                                      x * psrc1.nb[0]) /
                                     sizeof(float)];
                        val += bf16_to_fp32(ea) * bf16_to_fp32(fp32_to_bf16_round(eb));
                    }
                    res[i * h * m * n + j * m * n + p * n + l] = val;
                }
            }
        }
}

void matmul_e4m3q_f32_f32(std::vector<uint8_t> &a, std::vector<float> &b, std::vector<float> &res,
                          TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    int bs = psrc1.ne[3];
    int h = psrc1.ne[2];

    assert(psrc1.ne[3] == psrc0.ne[3]);
    assert(psrc1.ne[2] == psrc0.ne[2]);

    // src0 shape: (bs0, h0, n, k)
    // src1 shape: (bs, h, m, k)
    // dst shape: (bs, h, m, n)
    int n = psrc0.ne[1];
    int m = psrc1.ne[1];
    int k = psrc1.ne[0];
    assert(psrc1.ne[0] == psrc0.ne[0]);
    assert(k % QK == 0);
    // rowsize = sizeof block scales + sizeof block data
    int row_size = (k / QK) * sizeof(float) + k;

    for (int i = 0; i < bs; ++i)
        for (int j = 0; j < h; ++j) {
            for (int p = 0; p < m; ++p)
                for (int l = 0; l < n; ++l) {
                    float dot_sum = 0.0; // DOT SUM

                    // vector-dot loop
                    for (int ib = 0; ib < k / QK; ++ib) {
                        float block_sum = 0.0f; // BLOCK_SUM

                        // get A block (fp8)
                        block_e4m3_q *a_block_fp8 =
                            (block_e4m3_q *)(a.data() + i * h * n * row_size + j * n * row_size +
                                             l * row_size + ib * sizeof(block_e4m3_q));

                        // get B block (fp32)
                        float *b_block_f32 =
                            b.data() + i * h * m * k + j * m * k + p * k + (ib * QK);

                        // transfer B block (to fp8)
                        static struct block_e4m3_q b_block_fp8;
                        block_float_to_e4m3q(b_block_f32, &b_block_fp8);

                        // in-block loop
                        for (int x = 0; x < QK; ++x) {
                            float ea = e4m3q_to_float(a_block_fp8->data[x]);
                            float eb = e4m3q_to_float(b_block_fp8.data[x]);
                            block_sum += ea * eb;
                        }

                        // add block sum to dot sum (with scaling)
                        float scale_a = a_block_fp8->scale;
                        float scale_b = b_block_fp8.scale;
                        dot_sum += scale_a * scale_b * block_sum;
                    }

                    // vector-dot result
                    res[i * h * m * n + j * m * n + p * n + l] = dot_sum;
                }
        }
}

static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
int host_gmma_e4m3q_fp32_AT_QK256(void *a, void *b, void *c, int m, int n, int k, int lda, int ldb,
                                  int ldc) {
    printf("\nm:%d, n:%d, k:%d\n", m, n, k);
    printf("lda:%d, ldb:%d, ldc:%d\n", lda, ldb, ldc);
    static struct NPUKernelDesc *gmma_e4m3q_fp32_AT_QK256_desc = NULL;
    if (gmma_e4m3q_fp32_AT_QK256_desc == NULL) {
        ssize_t err = 0;
        printf("[INFO] Initializing Runtime..\n");
        err = npu_runtime_init(&runtime);
        if (err != NPU_SUCCESS) {
            printf("[ERROR] Failed to initialize Runtime: err=%d\n", err);
            exit(-1);
        }
        printf("[INFO] Runtime initialized successfully.\n");
        err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
        if (err != NPU_SUCCESS) {
            printf("Failed to load kernel set: err=%d\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Kernel set loaded successfully.\n");
        err = npu_runtime_get_kernel_desc(&runtime, "gmma_e4m3q_fp32_AT_QK256",
                                          &gmma_e4m3q_fp32_AT_QK256_desc);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%d\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded gmma_e4m3q_fp32_AT_QK256 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, m * lda * sizeof(uint8_t)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%d\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, n * ldb * sizeof(uint8_t)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%d\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, n * ldc * sizeof(float)); // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%d\n", c_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, gmma_e4m3q_fp32_AT_QK256_desc, &kernel_ret, 9, a_dev, b_dev,
                             c_dev, (uint64_t)m, (uint64_t)n, (uint64_t)k, (uint64_t)lda,
                             (uint64_t)ldb, (uint64_t)ldc);
    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, n * ldc * sizeof(float)); // c
    return kernel_ret;
}

void kernel_matmul_e4m3q(std::vector<uint8_t> &a, std::vector<float> &b, std::vector<float> &res,
                         TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    int bs = psrc1.ne[3];
    int h = psrc1.ne[2];
    int h0 = psrc0.ne[2];

    // dim(0) and dim(1) only broadcast from src0 to src1
    assert(bs >= psrc0.ne[3]);
    assert(h >= psrc0.ne[2]);
    assert(bs == 1);
    assert(h == 1);

    assert(pdst.nb[0] == sizeof(float));

    // src0 shape: (bs0, h0, m, k)
    // src1 shape: (bs, h, n, k)
    // dst shape: (bs, h, n, m)
    int m = psrc0.ne[1];
    int n = psrc1.ne[1];
    int k = psrc1.ne[0];
    assert(psrc1.ne[0] == psrc0.ne[0]);

    // Call Kernel
    host_gmma_e4m3q_fp32_AT_QK256(
        a.data(), b.data(), res.data(), m, n, k, psrc1.ne[0] * (QK + sizeof(float)) / QK,
        psrc0.ne[0] * (QK + sizeof(float)) / QK, pdst.nb[1] / sizeof(float));
}

size_t tensor_nbytes_bf16(const TensorParams &tensor) {
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] <= 0) {
            return 0;
        }
    }
    size_t nbytes;
    nbytes = sizeof(uint16_t);
    for (int i = 0; i < 4; ++i) {
        nbytes += (tensor.ne[i] - 1) * (tensor.nb[i]);
    }
    return nbytes;
}

size_t tensor_nbytes_fp32(const TensorParams &tensor) {
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] <= 0) {
            return 0;
        }
    }
    size_t nbytes;
    nbytes = sizeof(float);
    for (int i = 0; i < 4; ++i) {
        nbytes += (tensor.ne[i] - 1) * (tensor.nb[i]);
    }
    return nbytes;
}
int host_gmma_bf16_fp32_AT_QK256(void *a, void *b, void *c, int m, int n, int k, int lda, int ldb,
                                 int ldc, int g, int h, int stride_g, int stride_h, size_t nbytes_a,
                                 size_t nbytes_b, size_t nbytes_c) {
    printf("\nm:%d, n:%d, k:%d, g:%d, h:%d\n", m, n, k, g, h);
    printf("lda:%d, ldb:%d, ldc:%d, stride_g:%d, stride_h:%d\n", lda, ldb, ldc, stride_g, stride_h);
    for (int t = 0; t < 10; ++t) {
        printf("[%d] ", ((uint16_t *)a)[t]);
    }
    static struct NPUKernelDesc *gmma_bf16_fp32_AT_QK256_desc = NULL;
    if (gmma_bf16_fp32_AT_QK256_desc == NULL) {
        ssize_t err = 0;
        printf("[INFO] Initializing Runtime..\n");
        err = npu_runtime_init(&runtime);
        if (err != NPU_SUCCESS) {
            printf("[ERROR] Failed to initialize Runtime: err=%d\n", err);
            exit(-1);
        }
        printf("[INFO] Runtime initialized successfully.\n");
        err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
        if (err != NPU_SUCCESS) {
            printf("Failed to load kernel set: err=%d\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Kernel set loaded successfully.\n");
        err = npu_runtime_get_kernel_desc(&runtime, "gmma_bf16_fp32_AT_QK256",
                                          &gmma_bf16_fp32_AT_QK256_desc);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%d\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded gmma_bf16_fp32_AT_QK256 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, nbytes_a); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%d\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, nbytes_b); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%d\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, nbytes_c); // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%d\n", c_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, gmma_bf16_fp32_AT_QK256_desc, &kernel_ret, 13, a_dev, b_dev,
                             c_dev, (uint64_t)m, (uint64_t)n, (uint64_t)k, (uint64_t)lda,
                             (uint64_t)ldb, (uint64_t)ldc, (uint64_t)g, (uint64_t)h,
                             (uint64_t)stride_g, (uint64_t)stride_h);
    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, nbytes_c); // c
    return kernel_ret;
}

void kernel_matmul_bf16_f32_f32(std::vector<uint16_t> &a, std::vector<float> &b,
                                std::vector<float> &res, TensorParams &psrc0, TensorParams &psrc1,
                                TensorParams &pdst) {
    int bs = psrc1.ne[3];
    int h = psrc1.ne[2];
    int h0 = psrc0.ne[2];

    assert(bs >= psrc0.ne[3]);
    assert(h >= psrc0.ne[2]);
    assert(pdst.nb[0] == sizeof(float));

    // src0 shape: (bs0, h0, m, k)
    // src1 shape: (bs, h, n, k)
    // dst shape: (bs, h, n, m)
    int m = psrc0.ne[1];
    int n = psrc1.ne[1];
    int k = psrc1.ne[0];
    assert(psrc1.ne[0] == psrc0.ne[0]);

    // Call Kernel
    host_gmma_bf16_fp32_AT_QK256(
        a.data(), b.data(), res.data(), m, n, k, psrc0.nb[1] / sizeof(uint16_t),
        psrc1.nb[1] / sizeof(float), pdst.nb[1] / sizeof(float), psrc0.ne[2], psrc1.ne[2],
        psrc0.nb[2] / sizeof(uint16_t), psrc1.nb[2] / sizeof(float), tensor_nbytes_bf16(psrc0),
        tensor_nbytes_fp32(psrc1), tensor_nbytes_fp32(pdst));
}

int test_matmul(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    try {

        std::vector<float> data_b = readF32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

        std::vector<float> result_data;
        result_data.resize(psrc0.ne[1] * psrc1.ne[1] * psrc1.ne[2] * psrc1.ne[3]);

        if (psrc0.type == "bf16") {
            std::vector<uint16_t> data_a = readUInt16FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------
            kernel_matmul_bf16_f32_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);
            // --------------------------------------
            //
            // FIXME: CPU reference
            // matmul_bf16_f32_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);

        } else if (psrc0.type == "fp8_e4m3_q") {
            // TODO: pass
            // std::cout << "matmul fp8_e4m3_q is skipped" << std::endl;
            // return 0;
            std::vector<uint8_t> data_a = readUInt8FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------
            kernel_matmul_e4m3q(data_a, data_b, result_data, psrc0, psrc1, pdst);
            // --------------------------------------
            //
            // FIXME: CPU reference
            // matmul_e4m3q_f32_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);

        } else {
            std::cerr << "[ERROR] psrc0.dtype=" << psrc0.type << " is not supported\n" << std::endl;
            return 1;
        }

        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 0;
        bool all_match = true;

        int bs = pdst.ne[3];
        int h = pdst.ne[2];
        int n = pdst.ne[1];
        int m = pdst.ne[0];
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < n; ++k) {
                    for (int p = 0; p < m; ++p) {
                        float result = result_data[i * h * n * m + j * n * m + k * m + p];
                        float ref = data_ref[i * h * n * m + j * n * m + k * m + p];
                        if (std::abs((result - ref) / ref) > epsilon) {
                            std::cerr << std::fixed << std::setprecision(8);
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                            std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << result << std::endl;
                            std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << ref << std::endl;
                            std::cerr << "  - result - ref:        " << std::abs(result - ref)
                                      << std::endl;
                            std::cerr
                                << "  - reletive err:        " << std::abs((result - ref) / ref)
                                << std::endl;
                            all_match = false;
                            goto Done;
                        }
                    }
                }
            }
        }
    Done:
        if (all_match) {
            std::cout << " *** Compare PASS ***" << std::endl;
            return 0;
        } else
            return 1;

    } catch (const std::exception &e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./test <TRACE DIR> <TRACE ID>" << std::endl;
        return 0;
    }

    std::string trace_dir(argv[1]);
    int trace_id = std::stoi(argv[2]);

    TraceParamsParser parser;
    TensorParams psrc0, psrc1, pdst;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    std::string src1_trace = trace_dir + "/" + std::to_string(trace_id) + "-src1.trace";
    parser.parse(a, &pdst, &psrc0, &psrc1, NULL, NULL, NULL, NULL);
    if (!pdst.is_contiguous) {
        std::cerr << "[ERROR] matmul tests only support contiguous result tensor now!" << std::endl;
        return 1;
    }
    if ((psrc0.type != "bf16") && (psrc0.type != "fp8_e4m3_q")) {
        std::cerr << "[ERROR] matmul tests only support BF16/E4M3Q type in src0 tensor!"
                  << std::endl;
        return 1;
    }
    if ((psrc1.type != "f32") || (pdst.type != "f32")) {
        std::cerr << "[ERROR] matmul tests only support F32 type in src1/dst tensor!" << std::endl;
        return 1;
    }
    std::cout << "Testing Matmul with trace." << trace_id << " ------------------ ";
    return test_matmul(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst);
}