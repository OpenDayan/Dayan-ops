#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>

#include "common.hpp"

void rope_cache_init(std::vector<float>& cache,
                     float theta_base, float theta_scale,
                     float freq_scale,
                     int m,
                     float * corr_dims = NULL, // useless in qwen2
                     float * freq_factor = NULL, // freq_factor is always NULL in qwen2
                     float ext_factor = 0.0f, // // ext_factor is always 0.0f in qwen2
                     float mscale = 1.0f, // mscale == attn_factor is always 1.0f in qwen2
                     float sin_sign = 1.0f // sin_sign is always 1.0f in forward computing
                     ) {
    float theta = theta_base;
    for (size_t i = 0; i < m; i += 2) {
        // const float ff = freq_factors ? freq_factors[i/2] : 1.0f;
        const float ff = 1.0f;

        // YaRN
        float theta_extrap = theta / ff;
        float theta_interp = freq_scale * (theta / ff);
        // if (ext_factor != 0.0f) ...
        cache[i] = std::cos(theta_interp) * mscale;
        cache[i + 1] = std::sin(theta_interp) * mscale;

        cache[i + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

void rope_neox(std::vector<float>& a, std::vector<int>& b, std::vector<float>& res,
               TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst, RopeParams& prope) {
    int bs = psrc0.ne[3];
    int h = psrc0.ne[2];
    int n = psrc0.ne[1];
    int m = psrc0.ne[0];
    const float theta_scale = std::pow(prope.freq_base, -2.0f / prope.n_dims);

    std::vector<float> cache;
    cache.resize(psrc0.ne[0]);

    for (int i = 0; i < bs; ++i) { // batch
        for (int j = 0; j < h; ++j) { // seq-len
            int64_t p = b[j];
            rope_cache_init(cache, p, theta_scale,
                            static_cast<float>(prope.freq_scale), pdst.ne[0]);
            for (int k = 0; k < n; ++k) { // attn-heads
                // fill res[i][j][k][:prope.n_dims]
                for (int p = 0; p < prope.n_dims; p += 2) {
                    const int64_t ic = p / 2;
                    const float cos_theta = cache[p];
                    const float sin_theta = cache[p + 1];

                    const float x0 = a[i*h*n*m + j*n*m + k*m + ic]; // x0 = a[i][j][k][p]
                    const float x1 = a[i*h*n*m + j*n*m + k*m + ic + prope.n_dims / 2]; // x1 = a[i][j][k][p + dim/2]

                    res[i*h*n*m + j*n*m + k*m + ic] = (x0*cos_theta) - (x1*sin_theta); // dst[i][j][k][p] = ...
                    res[i*h*n*m + j*n*m + k*m + ic + prope.n_dims / 2] = (x0*sin_theta) + (x1*cos_theta); // dst[i][j][k][p + dim/2] = ...
                }

                // fill remain res[i][j][k][prope.n_dims:]
                for (size_t p = prope.n_dims; p < m; p += 2) {
                    res[i*h*n*m + j*n*m + k*m + p] = a[i*h*n*m + j*n*m + k*m + p];
                    res[i*h*n*m + j*n*m + k*m + p + 1] = a[i*h*n*m + j*n*m + k*m + p + 1];
                }
            }
        }
    }

}

int test_rope(std::string src0_trace, std::string src1_trace, std::string dst_trace,
             TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst, RopeParams& prope) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<int> data_b = readInt32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

        if (data_a.size() != data_ref.size()) {
            std::cerr << "Error: Input " << src0_trace << " has different size with " << dst_trace << std::endl;
            return 1;
        }

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        // std::cout << "\nTesting SiLU ..." << std::endl;
        std::vector<float> result_data;
        result_data.resize(data_a.size());

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        //
        // FIXME: CPU reference
        rope_neox(data_a, data_b, result_data, psrc0, psrc1, pdst, prope);

        // 对比结果
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
    TraceParamsParser parser;
    TensorParams psrc0, psrc1, pdst;
    RopeParams prope;
    std::cout << ">>>>>>>>>>>>> Start RoPE test:" << std::endl;
    for(int i = 0; i < 20; ++i) {
        std::string a = "trace/rope/" + std::to_string(i) + "-f32.txt";
        std::string dst_trace = "trace/rope/" + std::to_string(i) + "-f32-dst.trace";
        std::string src0_trace = "trace/rope/" + std::to_string(i) + "-f32-src0.trace";
        std::string src1_trace = "trace/rope/" + std::to_string(i) + "-i32-src1.trace";
        parser.parse(a, &pdst, &psrc0, &psrc1, NULL, &prope, NULL);
        if (!pdst.is_contiguous || !psrc0.is_contiguous || !psrc1.is_contiguous) {
            std::cerr << "[ERROR] RoPE tests only support contiguous tensor now!" << std::endl;
            return 1;
        }
        std::cout << "Testing RoPE with trace." << i << " ------------------ ";
        test_rope(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst, prope);
    }
    std::cout << "<<<<<<<<<<<<< RoPE test end" << std::endl;
}