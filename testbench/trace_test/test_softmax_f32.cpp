#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cstring>

#include "common.hpp"

void dumb_softmax(void) {}

void softmax(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res,
             TensorParams& pa, TensorParams& pb, SoftmaxParams& psoft) {
    const int bs          = pa.ne[3];
    const uint32_t h      = pa.ne[2];
    const int n           = pa.ne[1];
    const int m           = pa.ne[0];
    const uint32_t h_log2 = 1u << (uint32_t) floor(log2(h));

    const float m0 = powf(2.0f, -(psoft.max_bias       ) / h_log2);
    const float m1 = powf(2.0f, -(psoft.max_bias / 2.0f) / h_log2);

    for (int i = 0; i < bs; ++i) { // batch
        for (int j = 0; j < (h * n); ++j) { // attn-heads * seq_len
            const uint32_t hidx = (j / n); // current head idx
            // const float slope = (max_bias > 0.0f) ? h < h_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - h_log2) + 1) : 1.0f;
            const float slope = 1.0f; // only support max_bias == 0.0f in qwen2

            // scale
            // a[i][j][k][:] *= scale
            for (int p = 0; p < m; ++p) {
                a[i*h*n*m + j*m + p] = double(a[i*h*n*m + j*m + p]) * double(psoft.scale);
            }

            // mask
            // a[i][j][k][:] += slope * b[i][0][k][:]
            for (int p = 0; p < m; ++p) {
                a[i*h*n*m + j*m + p] += slope * b[i*h*n*m + (j%n)*m + p];
            }

            // get max value for dim[-1]
            // maxval = max(x(i))
            float maxval = -INFINITY;

            for (int p = 0; p < m; ++p) {
                if (a[i*h*n*m + j*m + p] > maxval)
                    maxval = a[i*h*n*m + j*m + p];
            }

            // a[i][j][k][:] -= max
            // x(i) = e^(x(i) - max)
            // sum = 1 / (sum of e^(x(i) - max))
            double sum = 0.0f;
            for (int p = 0; p < m; ++p) {
                float val = std::exp(a[i*h*n*m + j*m + p] - maxval);
                sum += (double)val;
                res[i*h*n*m + j*m + p] = val;
            }
            sum = 1.0 / sum;

            // res[i][j][k][:] *= 1/sum
            // x(i) = e^(x(i) - max) / (sum of e^(x(i) - max))
            for (int p = 0; p < m; ++p) {
                res[i*h*n*m + j*m + p] *= (float)sum;
            }
        }
    }
}

int test_softmax(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                 TensorParams & psrc0, TensorParams & psrc1, TensorParams & pdst, SoftmaxParams & psoft) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<float> data_b = readF32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

        if (data_a.size() != data_ref.size()) {
            std::cerr << "Error: Input " << src0_trace << " has different size with " << dst_trace << std::endl;
            return 1;
        }

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        std::vector<float> result_data;
        result_data.resize(data_a.size());

        // FIXME: Run kernel here
        //
        // --------------------------------------------
        // kernel_sofmax(data_a, data_b, result_data, ...);
        // --------------------------------------------
        //
        // FIXME: CPU reference (should be replaced)
        softmax(data_a, data_b, result_data, psrc0, psrc1, psoft);

        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 1e-6f;
        bool all_match = true;

        int bs = psrc0.ne[3];
        int h = psrc0.ne[2];
        int n = psrc0.ne[1];
        int m = psrc0.ne[0];
        for (size_t i = 0; i < bs; ++i) {
            for (size_t j = 0; j < h; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    for (size_t p = 0; p < m; ++p) {
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
    TensorParams psrc0, psrc1, pdst;
    SoftmaxParams psoft;
    std::cout << ">>>>>>>>>>>>> Start Softmax test:" << std::endl;
    for(int i = 0; i < test_num; ++i) {
        std::string a = "trace/softmax/" + std::to_string(i) + "-params.txt";
        std::string dst_trace = "trace/softmax/" + std::to_string(i) + "-dst.trace";
        std::string src0_trace = "trace/softmax/" + std::to_string(i) + "-src0.trace";
        std::string src1_trace = "trace/softmax/" + std::to_string(i) + "-src1.trace";
        parser.parse(a, &pdst, &psrc0, &psrc1, NULL, NULL, &psoft);
        if (psoft.max_bias != 0.0f) {
            std::cerr << "[ERROR] Softmax tests only support max_bias == 0.0f" << std::endl;
            return 1;
        }
        std::cout << "Testing Softmax with trace." << i << " ------------------ ";
        test_softmax(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst, psoft);
    }
    std::cout << "<<<<<<<<<<<<< Softmax test end" << std::endl;
}