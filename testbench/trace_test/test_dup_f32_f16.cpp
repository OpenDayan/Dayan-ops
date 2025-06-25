#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <bit>
#include <cstring>
#include "common.hpp"

uint16_t float_to_fp16(float value) {
    uint32_t fbits = 0;
    std::memcpy(&fbits, &value, sizeof(float));
    unsigned int sign = (fbits >> 16) & 0x8000;
    fbits &= 0x7FFFFFFF;
    if (fbits >= 0x7F800000)
        return sign | 0x7C00 | ((fbits > 0x7F800000) ? (0x200 | ((fbits >> 13) & 0x3FF)) : 0);
    if (fbits >= 0x47800000)
        return sign | 0x7c00;
    if (fbits >= 0x38800000) {
        uint32_t value = sign | (((fbits >> 23) - 112) << 10) | ((fbits >> 13) & 0x3FF);
        uint32_t guard = (fbits >> 12) & 1;
        uint32_t sticky = (fbits & 0xFFF) != 0;
        return value + (guard & (sticky | value));
    }
    if (fbits >= 0x33000000) {
        int i = 125 - (fbits>>23);
        fbits = (fbits & 0x7FFFFF) | 0x800000;
        uint32_t value = sign | (fbits >> (i + 1));
        uint32_t guard = (fbits >> i) & 1;
        uint32_t sticky = (fbits & ((static_cast<uint32_t>(1) << i) - 1)) != 0;
        return value + (guard & (sticky | value));
    }
    if (fbits != 0)
        return sign;
    return sign;
}

void dup_no_contiguous(std::vector<float>& a, std::vector<uint16_t>& res, TensorParams& pa, TensorParams& pres) {
    if((pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3])) {
        std::cerr << "[ERROR] Non-contiguous tensor has different shape" << std::endl;
        return;
    }
    int bs = pa.ne[3];
    int h = pa.ne[2];
    int n = pa.ne[1];
    int m = pa.ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    res[i*pres.nb[3]/sizeof(uint16_t) +
                        j*pres.nb[2]/sizeof(uint16_t) +
                        k*pres.nb[1]/sizeof(uint16_t) +
                        p*pres.nb[0]/sizeof(uint16_t)] =
                            float_to_fp16(a[i*pa.nb[3]/sizeof(float) +
                                                  j*pa.nb[2]/sizeof(float) +
                                                  k*pa.nb[1]/sizeof(float) +
                                                  p*pa.nb[0]/sizeof(float)]);
                }
            }
        }
    }
}

void dup_contiguous(std::vector<float>& a, std::vector<uint16_t>& res, TensorParams& pa, TensorParams& pres) {
    for (size_t i = 0; i < a.size(); ++i) {
        res[i] = float_to_fp16(a[i]);
    }
}

int test_dup(std::string src0_trace, std::string dst_trace, TensorParams & psrc0, TensorParams & pdst) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<uint16_t> data_ref = readUInt16FromFile(dst_trace);

        size_t len_src0 = psrc0.ne[0] * psrc0.ne[1] * psrc0.ne[2] * psrc0.ne[3];
        size_t len_dst = pdst.ne[0] * pdst.ne[1] * pdst.ne[2] * pdst.ne[3];
        if (len_src0 != len_dst) {
            std::cerr << "Error: Input " << src0_trace << " has different shape with " << dst_trace << std::endl;
            return 1;
        }

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        std::vector<uint16_t> result_data;
        result_data.resize(data_ref.size());

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // kernel_dup_f32_f16(data_a, result_data, ...);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        if (psrc0.is_contiguous)
            dup_contiguous(data_a, result_data, psrc0, pdst);
        else
            dup_no_contiguous(data_a, result_data, psrc0, pdst);

        // std::cout << "\nCompare with ref..." << std::endl;
        bool all_match = true;

        if (psrc0.is_contiguous) {
            for (size_t i = 0; i < result_data.size(); ++i) {
                if (result_data[i] != data_ref[i]) {
                    std::cerr << std::fixed << std::setprecision(8);
                    std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                    std::cerr << "  - input0[ ... ][" << i % pdst.ne[0] << "]:    " << std::hex << data_a[i] << std::dec << std::endl;
                    std::cerr << "  - result[ ... ][" << i % pdst.ne[0] << "]:    " << std::hex << result_data[i] << std::dec << std::endl;
                    std::cerr << "  - ref[ ... ][" << i % pdst.ne[0] << "]:    " << std::hex << data_ref[i] << std::dec << std::endl;
                    all_match = false;
                    break;
                }
            }
        } else {
            int bs = pdst.ne[3];
            int h = pdst.ne[2];
            int n = pdst.ne[1];
            int m = pdst.ne[0];
            for (int i = 0; i < bs; ++i) {
                for (int j = 0; j < h; ++j) {
                    for (int k = 0; k < n; ++k) {
                        for (int p = 0; p < m; ++p) {
                            uint16_t ref = data_ref[i*pdst.nb[3]/sizeof(uint16_t) +
                                                    j*pdst.nb[2]/sizeof(uint16_t) +
                                                    k*pdst.nb[1]/sizeof(uint16_t) +
                                                    p*pdst.nb[0]/sizeof(uint16_t)];
                            uint16_t result = result_data[i*pdst.nb[3]/sizeof(uint16_t) +
                                                          j*pdst.nb[2]/sizeof(uint16_t) +
                                                          k*pdst.nb[1]/sizeof(uint16_t) +
                                                          p*pdst.nb[0]/sizeof(uint16_t)];
                            if (ref != result) {
                                std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                                std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p << "]:    "
                                          << std::hex << result << std::dec << std::endl;
                                std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p << "]:    "
                                          << std::hex << ref << std::dec << std::endl;
                                all_match = false;
                                goto Done;
                            }
                        }
                    }
                }
            }
        }
        Done:
        if (all_match) {
            std::cout << " *** Compare PASS ***" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

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
    TensorParams psrc0, pdst;
    std::cout << ">>>>>>>>>>>>> Start Add test:" << std::endl;
    for(int i = 0; i < test_num; ++i) {
        std::string a = "trace/dup/" + std::to_string(i) + "-params.txt";
        std::string dst_trace = "trace/dup/" + std::to_string(i) + "-dst.trace";
        std::string src0_trace = "trace/dup/" + std::to_string(i) + "-src0.trace";
        parser.parse(a, &pdst, &psrc0, NULL, NULL, NULL, NULL);
        if((psrc0.is_contiguous != pdst.is_contiguous) ||
           (psrc0.type != "f32") ||
           (pdst.type != "f16")) {
            std::cerr << "[ERROR] Dup tests only support FP32 to FP16 test and src0.is_contiguous == dst.is_contiguous" << std::endl;
            return 1;
        }
        std::cout << "Testing Dup with trace." << i << " ------------------ ";
        test_dup(src0_trace, dst_trace, psrc0, pdst);
    }
    std::cout << "<<<<<<<<<<<<< Dup test end" << std::endl;
}