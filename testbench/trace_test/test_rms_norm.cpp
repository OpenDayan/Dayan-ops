#include <cstddef>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>

#include "common.hpp"

void rms_norm(std::vector<float>& a, std::vector<float>& res,
              TensorParams & psrc0, NormParams & pnorm) {
    int b = psrc0.ne[3];
    int h = psrc0.ne[2];
    int n = psrc0.ne[1];
    int m = psrc0.ne[0];
    for (int i = 0; i < (b * h * n); ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += (double)(a[i * m + j] * a[i * m + j]);
        }
        float mean = (float)(sum / m);
        for (int j = 0; j < m; ++j) {
            res[i * m + j] = a[i * m + j];
            res[i * m + j] *= 1.0f/std::sqrt(mean + (float)(pnorm.eps));
        }
    }
}

int test_rms_norm(std::string src0_trace, std::string dst_trace,
                  TensorParams & psrc0, TensorParams & pdst, NormParams & pnorm) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
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

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // kernel_rms_norm(data_a, result_data, ...);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        rms_norm(data_a, result_data, psrc0, pnorm);

        // 对比结果
        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 1e-6f;
        bool all_match = true;

        for (size_t i = 0; i < result_data.size(); ++i) {
            if (std::abs(result_data[i] - data_ref[i]) > epsilon) {
                std::cerr << std::fixed << std::setprecision(8);
                std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                std::cerr << "  - input1[ ... ][" << i % pdst.ne[0] << "]:    " << data_a[i] << std::endl;
                std::cerr << "  - result[ ... ][" << i % pdst.ne[0] << "]:    " << result_data[i] << std::endl;
                std::cerr << "  - ref[" << i / pdst.ne[0] << "][" << i % pdst.ne[0] << "]:    " << data_ref[i] << std::endl;
                std::cerr << "  - result - ref:        " << std::abs(result_data[i] - data_ref[i]) << std::endl;
                all_match = false;
                break;
            }
        }

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
    NormParams pnorm;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    parser.parse(a, &pdst, &psrc0, NULL, &pnorm, NULL, NULL);
    if (!pdst.is_contiguous || !psrc0.is_contiguous) {
        std::cerr << "[ERROR] RMS_Norm tests only support contiguous tensor now!" << std::endl;
        return 1;
    }
    std::cout << "Testing RMS_Norm with trace." << trace_id << " ------------------ ";
    return test_rms_norm(src0_trace, dst_trace, psrc0, pdst, pnorm);
}