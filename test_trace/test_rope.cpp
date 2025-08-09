#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cassert>

#include <NPURuntime.h>

#include "common.hpp"

void rope_cache_init(std::vector<float>& cache,
                     float theta_base, float theta_scale,
                     float freq_scale,
                     int m,
                     float* corr_dims = NULL, // useless in qwen2
                     float* freq_factor = NULL, // freq_factor is always NULL in qwen2
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
                    const float x0 = a[i * h * n * m + j * n * m + k * m + ic];                    // x0 = a[i][j][k][p]
                    const float x1 = a[i * h * n * m + j * n * m + k * m + ic + prope.n_dims / 2]; // x1 = a[i][j][k][p + dim/2]

                    res[i * h * n * m + j * n * m + k * m + ic] = (x0 * cos_theta) - (x1 * sin_theta);                    // dst[i][j][k][p] = ...
                    res[i * h * n * m + j * n * m + k * m + ic + prope.n_dims / 2] = (x0 * sin_theta) + (x1 * cos_theta); // dst[i][j][k][p + dim/2] = ...
                }

                // fill remain res[i][j][k][prope.n_dims:]
                for (size_t p = prope.n_dims; p < m; p += 2) {
                    res[i * h * n * m + j * n * m + k * m + p] = a[i * h * n * m + j * n * m + k * m + p];
                    res[i * h * n * m + j * n * m + k * m + p + 1] = a[i * h * n * m + j * n * m + k * m + p + 1];
                }
            }
        }
    }
}


static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
int host_rope_f32_i32_f32(
    void* a, void* b, void* res,
    int bs, int h, int n, int m, int n_dims,
    float theta_scale, float freq_scale, int mdst) {
    static struct NPUKernelDesc* rope_f32_i32_f32 = NULL;
    if (rope_f32_i32_f32 == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "rope_f32_i32_f32", &rope_f32_i32_f32);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%d\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded rope_f32_i32_f32 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, bs * h * n * m * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%d\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, h * sizeof(int)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%d\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res, bs * h * n * m * sizeof(float));   // res
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%d\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, rope_f32_i32_f32, &kernel_ret, 11,
                             a_dev, b_dev, res_dev,
                             (uint64_t)bs, (uint64_t)h, (uint64_t)n, (uint64_t)m, (uint64_t)n_dims,
                             (float)theta_scale, (float)freq_scale, (int)mdst);
    npu_memcpy_from_dev(&runtime.dev_state, res, res_dev, bs * h * n * m * sizeof(float)); // res
    return kernel_ret;
}

void kernel_rope_f32(std::vector<float>& a, std::vector<int>& b, std::vector<float>& res,
                     TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst, RopeParams& prope) {
    int bs = psrc0.ne[3];
    int h = psrc0.ne[2];
    int n = psrc0.ne[1];
    int m = psrc0.ne[0];
    float theta_scale = std::pow(prope.freq_base, -2.0f / prope.n_dims);

    host_rope_f32_i32_f32(
        a.data(),
        b.data(),
        res.data(),
        bs, h, n, m, prope.n_dims,
        theta_scale, prope.freq_scale, pdst.ne[0]
    );
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

        std::vector<float> result_data;
        result_data.resize(data_a.size());

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        kernel_rope_f32(data_a, data_b, result_data, psrc0, psrc1, pdst, prope);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        // rope_neox(data_a, data_b, result_data, psrc0, psrc1, pdst, prope);

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
                        float result = result_data[i * h * n * m + j * n * m + k * m + p];
                        float ref = data_ref[i * h * n * m + j * n * m + k * m + p];

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
            return 0;
        }
        else
            return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./test <TRACE DIR> <TRACE ID>" << std::endl;
        return 0;
    }

    std::string trace_dir(argv[1]);
    int trace_id = std::stoi(argv[2]);

    TraceParamsParser parser;
    TensorParams psrc0, psrc1, pdst;
    RopeParams prope;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    std::string src1_trace = trace_dir + "/" + std::to_string(trace_id) + "-src1.trace";
    parser.parse(a, &pdst, &psrc0, &psrc1, NULL, &prope, NULL, NULL);
    if (!pdst.is_contiguous || !psrc0.is_contiguous || !psrc1.is_contiguous) {
        std::cerr << "[ERROR] RoPE tests only support contiguous tensor now!" << std::endl;
        return 1;
    }
    std::cout << "Testing RoPE with trace." << trace_id << " ------------------ ";
    return test_rope(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst, prope);
}