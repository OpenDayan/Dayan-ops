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

// contiguous f32 copy to contiguous f316 (in prefill/decode stage)
void dup_cont_f32_cont_f16(std::vector<float>& a, std::vector<uint16_t>& res, TensorParams& pa, TensorParams& pres) {
    if((pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3])) {
        std::cerr << "[ERROR] Contiguous fp32 and fp16 tensor has different shape" << std::endl;
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
                    res[i*h*n*m + j*n*m + k*m + p] = float_to_fp16(a[i*h*n*m + j*n*m + k*m + p]);
                }
            }
        }
    }
}

// non-contiguous f32 copy to non-contiguous f16 (in decode stage)
void dup_cont_f32_no_cont_f16(std::vector<float>& a, std::vector<uint16_t>& res, TensorParams& pa, TensorParams& pres) {
    if((pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3])) {
        std::cerr << "[ERROR] Contiguous fp32 and non-contiguous fp16 tensor has different shape" << std::endl;
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
                            float_to_fp16(a[i*h*n*m + j*n*m + k*m + p]);
                }
            }
        }
    }
}

// contiguous f32 copy to non-contiguous f16 (in prefill/decode stage)
void dup_no_cont_f32_no_cont_f16(std::vector<float>& a, std::vector<uint16_t>& res, TensorParams& pa, TensorParams& pres) {
    if((pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3])) {
        std::cerr << "[ERROR] Non-contiguous fp32 and fp 16 tensor has different shape" << std::endl;
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

// contiguous f32 copy to contiguous f32 (in decode stage)
void dup_cont_f32_cont_f32(std::vector<float>& a, std::vector<float>& res, TensorParams& pa, TensorParams& pres) {
    if((pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3])) {
        std::cerr << "[ERROR] Contiguous fp32 and fp32 tensor has different shape" << std::endl;
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
                    res[i*h*n*m + j*n*m + k*m + p] = a[i*h*n*m + j*n*m + k*m + p];
                }
            }
        }
    }
}

// non-contiguous f32 copy to contiguous f32 (in prefill stage)
void dup_no_cont_f32_cont_f32(std::vector<float>& a, std::vector<float>& res, TensorParams& pa, TensorParams& pres) {
    if((pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3]) ||
       (pa.ne[3] != pres.ne[3])) {
        std::cerr << "[ERROR] Non-contiguous fp32 and contiguous fp32 tensor has different shape" << std::endl;
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
                    res[i*h*n*m + j*n*m + k*m + p] = a[i*pa.nb[3]/sizeof(float) +
                                                       j*pa.nb[2]/sizeof(float) +
                                                       k*pa.nb[1]/sizeof(float) +
                                                       p*pa.nb[0]/sizeof(float)];
                }
            }
        }
    }
}



int test_dup_f32_to_f16(std::string src0_trace, std::string dst_trace, TensorParams & psrc0, TensorParams & pdst) {
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
        // if (pdst.is_contiguous)
        //     kernel_dup_cont_f16(data_a, result_data, ...);
        // else
        //     kernel_dup_no_cont_f16(data_a, result_data, ...);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        if (pdst.is_contiguous && psrc0.is_contiguous)
            dup_cont_f32_cont_f16(data_a, result_data, psrc0, pdst);
        else if (!pdst.is_contiguous && !psrc0.is_contiguous)
            dup_no_cont_f32_no_cont_f16(data_a, result_data, psrc0, pdst);
        else if (psrc0.is_contiguous && !pdst.is_contiguous)
            dup_cont_f32_no_cont_f16(data_a, result_data, psrc0, pdst);
        else {
            std::cerr << "Not supported src0.is_contiguous=" << psrc0.is_contiguous <<
                         ", dst.is_contiguous=" << pdst.is_contiguous << std::endl;
            return 1;
        }

        // std::cout << "\nCompare with ref..." << std::endl;
        bool all_match = true;

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
        Done:
        if (all_match) {
            std::cout << " *** Compare PASS ***" << std::endl;
            return 0;
        } else
            return 1;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int test_dup_f32_to_f32(std::string src0_trace, std::string dst_trace, TensorParams & psrc0, TensorParams & pdst) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

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

        std::vector<float> result_data;
        result_data.resize(data_ref.size());

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // kernel_dup_no_cont_f32_cont_f32(data_a, result_data, ...);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        if (psrc0.is_contiguous && pdst.is_contiguous)
            dup_cont_f32_cont_f32(data_a, result_data, psrc0, pdst);
        else if (!psrc0.is_contiguous && pdst.is_contiguous)
            dup_no_cont_f32_cont_f32(data_a, result_data, psrc0, pdst);
        else {
            std::cerr << "Not supported src0.is_contiguous=" << psrc0.is_contiguous <<
                         ", dst.is_contiguous=" << pdst.is_contiguous << std::endl;
            return 1;
        }

        // std::cout << "\nCompare with ref..." << std::endl;
        bool all_match = true;

        int bs = pdst.ne[3];
        int h = pdst.ne[2];
        int n = pdst.ne[1];
        int m = pdst.ne[0];
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < n; ++k) {
                    for (int p = 0; p < m; ++p) {
                        float ref = data_ref[i*h*n*m + j*n*m + k*m + p];
                        float result = result_data[i*h*n*m + j*n*m + k*m + p];
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
        Done:
        if (all_match) {
            std::cout << " *** Compare PASS ***" << std::endl;
            return 0;
        } else
            return 1;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}



int main(int argc, char * argv[]) {
    if(argc < 3) {
        std::cout << "Usage: ./test <TRACE DIR> <TRACE ID>" << std::endl;
        return 0;
    }

    std::string trace_dir(argv[1]);
    int trace_id = std::stoi(argv[2]);

    TraceParamsParser parser;
    TensorParams psrc0, pdst;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    parser.parse(a, &pdst, &psrc0, NULL, NULL, NULL, NULL);
    if (!(psrc0.type == "f32" && pdst.type == "f32") &&
        !(psrc0.type == "f32" && pdst.type == "f16")) {
        std::cerr << "[ERROR] Dup tests only support FP32 to FP16/Fp32 test" << std::endl;
        return 1;
    }
    // dup f32 to f16
    if ((psrc0.type == "f32" && pdst.type == "f16") &&
        (!psrc0.is_contiguous && pdst.is_contiguous)) {
        std::cerr << "[ERROR] Dup tests does not support non-contiguous src0 and contiguous dst for FP32 to FP16 test" << std::endl;
        return 1;
    }
    // dup f32 to f32 输出张量连续
    if ((psrc0.type == "f32" && pdst.type == "f32") &&
        !pdst.is_contiguous) {
        std::cerr << "[ERROR] Dup tests only support contiguous dst for FP32 to FP32 test" << std::endl;
        return 1;
    }
    std::cout << "Testing Dup with trace." << trace_id << " ------------------ ";
    if (pdst.type == "f16")
        return test_dup_f32_to_f16(src0_trace, dst_trace, psrc0, pdst);
    return test_dup_f32_to_f32(src0_trace, dst_trace, psrc0, pdst);
}