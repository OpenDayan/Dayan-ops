#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <x86intrin.h>

#include "common.hpp"

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
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
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

float e4m3q_to_float(uint8_t h) {
    uint32_t out_u32 = 0;
    float out_f32 = 0.0f;
    out_u32 = h & 0x80;
    out_u32 <<= 24;
    uint32_t h_bits = h & 0x7F;
    h_bits <<= (23 - 3); // 23 - FP8::M (M==3 in e4m3q)
    out_u32 |= h_bits;
    // 2^(127 - FP8::E_BIAS)
    out_f32 *= std::pow(2, 127 - 7); // E==4 in e4m3q, so E_BIAS = 2^(E-1) - 1 = 7
    return out_f32;
}

void matmul_f16_f32_f32(std::vector<uint16_t>& a, std::vector<float>& b, std::vector<float>& res,
                        TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
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
            for (int p = 0; p < m; ++p)
                for (int l = 0; l < n; ++l) {
                    double val = 0.0;
                    // res[i][j][p][l] = b[i][j][p][:] * a[i0][j0][l][:]^T
                    for (int x = 0; x < k; ++x) {
                        uint16_t ea = a[(i0*psrc0.nb[3] + j0*psrc0.nb[2] + l*psrc0.nb[1] + x*psrc0.nb[0]) / sizeof(uint16_t)];
                        float eb = b[(i*psrc1.nb[3] + j*psrc1.nb[2] + p*psrc1.nb[1] + x*psrc1.nb[0]) / sizeof(float)];
                        // b[...] is converted to fp16 then converted to fp32
                        val += (double)(fp16_to_fp32(ea) * fp16_to_fp32(fp32_to_fp16(eb)));
                    }
                    // double to float cast
                    res[i*h*m*n + j*m*n + p*n + l] = val;
                }
        }
}

void matmul_e4m3q_f32_f32(std::vector<uint8_t>& a, std::vector<float>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
    int bs = psrc1.ne[3];
    int h = psrc1.ne[2];

    // dim(0) and dim(1) only broadcast from src0 to src1
    int rbs = psrc1.ne[3] / psrc0.ne[3];
    int rj = psrc1.ne[2] / psrc0.ne[2];
    assert(psrc1.ne[3] >= psrc0.ne[3]);
    assert(psrc1.ne[2] >= psrc0.ne[2]);

    // src0 shape: (bs0, h0, n, k)
    // src1 shape: (bs, h, m, k)
    // dst shape: (bs, h, m, n)
    int n = psrc0.ne[1];
    int m = psrc1.ne[1];
    int k = psrc1.ne[0];
    assert(psrc1.ne[0] == psrc0.ne[0]);

    #define QK 256
    assert(k % QK == 0);

    for (int i = 0; i < bs; ++i)
        for (int j = 0; j < h; ++j) {
            // broadcast in dim(0)
            int i0 = i / rbs;
            // broadcase in dim(1)
            int j0 = j / rj;

            // b[i][j][:][:] * a[i0][j0][:][:]^T
            for (int p = 0; p < m; ++p)
                for (int l = 0; l < n; ++l) {
                    float val = 0.0;
                    // res[i][j][p][l] += b[i][j][p][:] * a[i0][j0][l][:]^T
                    for (int ib = 0; ib < k/QK; ++ib) {
                        float sum = 0.0f;
                        for (int x = 0; x < QK; ++x) {
                            // TODO:
                        }
                        // TODO: val += sum * scale
                    }
                    res[i*h*m*n + j*m*n + p*n + l] = val;
                }
        }
}

int test_matmul(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
    // try {

        std::vector<float> data_b = readF32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

        std::vector<float> result_data;
        result_data.resize(psrc0.ne[1] * psrc1.ne[1] * psrc1.ne[2] * psrc1.ne[3]);

        if (psrc0.type == "f16") {
            std::vector<uint16_t> data_a = readUInt16FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------
            // kernel_matmul_f16_f32_f32(data_a, data_b, result_data, ...);
            // --------------------------------------
            //
            // FIXME: CPU reference
            matmul_f16_f32_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);

        } else if (psrc0.type == "fp8_e4m3_q") {
            // TODO: pass
            std::cout << "matmul fp8_e4m3_q is skipped" << std::endl;
            return 0;
            std::vector<uint8_t> data_a = readUInt8FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------
            // kernel_matmul_e4m3q(data_a, data_b, result_data, ...);
            // --------------------------------------
            //
            // FIXME: CPU reference
            matmul_e4m3q_f32_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);

        } else {
            std::cerr << "[ERROR] psrc0.dtype=" << psrc0.type << " is not supported\n" << std::endl;
            return 1;
        }

        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 1e-6f;
        bool all_match = true;

        int bs = pdst.ne[3];
        int h = pdst.ne[2];
        int n = pdst.ne[1];
        int m = pdst.ne[0];
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < n; ++k) {
                    for (int p = 0; p < m; ++p) {
                        float result = result_data[i*h*n*m + j*n*m + k*m + p];
                        float ref = data_ref[i*h*n*m + j*n*m + k*m + p];
                        if (std::abs(result - ref) > epsilon) {
                            std::cerr << std::fixed << std::setprecision(8);
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                            std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p << "]:    "
                                    << result << std::endl;
                            std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p << "]:    "
                                    << ref << std::endl;
                            std::cerr << "  - result - ref:        " << std::abs(result - ref) << std::endl;
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
        }

    // } catch (const std::exception& e) {
    //     std::cerr << "FATAL ERROR: " << e.what() << std::endl;
    //     return 1;
    // }

    return 0;

}

int main(int argc, char * argv[]) {
    int test_num = 0;
    if(argc < 2) {
        std::cout << "Usage: ./test <TRACE NUM>" << std::endl;
        return 0;
    } else {
        test_num = std::stoi(argv[1]);
        if (test_num > 1000) {
            std::cerr << "[ERROR] Trace num " << test_num << "exceeds limit (1000)" << std::endl;
            return 1;
        }
    }
    TraceParamsParser parser;
    TensorParams psrc0, psrc1, pdst;
    std::cout << ">>>>>>>>>>>>> Start Matmul test:" << std::endl;
    for(int i = 0; i < test_num; ++i) {
        std::string a = "trace/matmul/" + std::to_string(i) + "-params.txt";
        std::string dst_trace = "trace/matmul/" + std::to_string(i) + "-dst.trace";
        std::string src0_trace = "trace/matmul/" + std::to_string(i) + "-src0.trace";
        std::string src1_trace = "trace/matmul/" + std::to_string(i) + "-src1.trace";
        parser.parse(a, &pdst, &psrc0, &psrc1, NULL, NULL, NULL);
        if (!pdst.is_contiguous) {
            std::cerr << "[ERROR] matmul tests only support contiguous result tensor now!" << std::endl;
            return 1;
        }
        if ((psrc0.type != "f16") && (psrc0.type != "fp8_e4m3_q")) {
            std::cerr << "[ERROR] matmul tests only support FP16/E4M3Q type in src0 tensor!" << std::endl;
            return 1;
        }
        if ((psrc1.type != "f32") || (pdst.type != "f32")) {
            std::cerr << "[ERROR] matmul tests only support F32 type in src1/dst tensor!" << std::endl;
            return 1;
        }
        std::cout << "Testing Matmul with trace." << i << " ------------------ ";
        test_matmul(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst);
    }
    std::cout << "<<<<<<<<<<<<< Matmul test end" << std::endl;
}