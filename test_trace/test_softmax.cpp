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

#include <NPURuntime.h>

#include "common.hpp"

void dumb_softmax(void) {}

void softmax(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res,
             TensorParams& pa, TensorParams& pb, SoftmaxParams& psoft) {
    const int bs = pa.ne[3];
    const uint32_t h = pa.ne[2];
    const int n = pa.ne[1];
    const int m = pa.ne[0];
    const uint32_t h_log2 = 1u << (uint32_t)floor(log2(h));

    const float m0 = powf(2.0f, -(psoft.max_bias) / h_log2);
    const float m1 = powf(2.0f, -(psoft.max_bias / 2.0f) / h_log2);

    for (int i = 0; i < bs; ++i) { // batch
        for (int j = 0; j < (h * n); ++j) { // attn-heads * seq_len
            const uint32_t hidx = (j / n); // current head idx
            // const float slope = (max_bias > 0.0f) ? h < h_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - h_log2) + 1) : 1.0f;
            const float slope = 1.0f; // only support max_bias == 0.0f in qwen2

            // scale
            // a[i][j][k][:] *= scale
            for (int p = 0; p < m; ++p) {
                a[i * h * n * m + j * m + p] = double(a[i * h * n * m + j * m + p]) * double(psoft.scale);
            }

            // mask
            // a[i][j][k][:] += slope * b[i][0][k][:]
            for (int p = 0; p < m; ++p) {
                a[i * h * n * m + j * m + p] += slope * b[i * h * n * m + (j % n) * m + p];
            }

            // get max value for dim[-1]
            // maxval = max(x(i))
            float maxval = -INFINITY;

            for (int p = 0; p < m; ++p) {
                if (a[i * h * n * m + j * m + p] > maxval)
                    maxval = a[i * h * n * m + j * m + p];
            }

            // a[i][j][k][:] -= max
            // x(i) = e^(x(i) - max)
            // sum = 1 / (sum of e^(x(i) - max))
            double sum = 0.0f;
            for (int p = 0; p < m; ++p) {
                float val = std::exp(a[i * h * n * m + j * m + p] - maxval);
                sum += (double)val;
                res[i * h * n * m + j * m + p] = val;
            }
            sum = 1.0 / sum;

            // res[i][j][k][:] *= 1/sum
            // x(i) = e^(x(i) - max) / (sum of e^(x(i) - max))
            for (int p = 0; p < m; ++p) {
                res[i * h * n * m + j * m + p] *= (float)sum;
            }
        }
    }
}

static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
int host_softmax_f32_f32_f32(float* a, float* b, float* res, float scale,
              int bs, int h, int n, int m) {
    static struct NPUKernelDesc* softmax_f32_f32_f32 = NULL;
    if (softmax_f32_f32_f32 == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "softmax_f32_f32_f32", &softmax_f32_f32_f32);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%d\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded softmax_f32_f32_f32 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, bs * h * n * m * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write x to device at 0x%lx: err=%d\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, bs * h * n * m * sizeof(float)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write pos_sin to device at 0x%lx: err=%d\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res, bs * h * n * m * sizeof(float));   // res
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%d\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, softmax_f32_f32_f32, &kernel_ret, 8,
                             a_dev, b_dev, res_dev, (float)scale,
                             (uint64_t)bs, (uint64_t)h, (uint64_t)n, (uint64_t)m);
    npu_memcpy_from_dev(&runtime.dev_state, res, res_dev, bs * h * n * m * sizeof(float)); // res
    return kernel_ret;
}

void kernel_softmax(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res,
             TensorParams& pa, TensorParams& pb, SoftmaxParams& psoft) {
    const int bs = pa.ne[3];
    const uint32_t h = pa.ne[2];
    const int n = pa.ne[1];
    const int m = pa.ne[0];
    const uint32_t h_log2 = 1u << (uint32_t)floor(log2(h));

    const float m0 = powf(2.0f, -(psoft.max_bias) / h_log2);
    const float m1 = powf(2.0f, -(psoft.max_bias / 2.0f) / h_log2);

    const float scale = psoft.scale;

    host_softmax_f32_f32_f32(a.data(), b.data(), res.data(), scale, bs, h, n, m);
}

int test_softmax(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                 TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst, SoftmaxParams& psoft) {
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
        kernel_softmax(data_a, data_b, result_data, psrc0, psrc1, psoft);
        // --------------------------------------------
        //
        // FIXME: CPU reference (should be replaced)
        // softmax(data_a, data_b, result_data, psrc0, psrc1, psoft);

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
    SoftmaxParams psoft;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    std::string src1_trace = trace_dir + "/" + std::to_string(trace_id) + "-src1.trace";
    parser.parse(a, &pdst, &psrc0, &psrc1, NULL, NULL, &psoft, NULL);
    if (psoft.max_bias != 0.0f) {
        std::cerr << "[ERROR] Softmax tests only support max_bias == 0.0f" << std::endl;
        return 1;
    }
    std::cout << "Testing Softmax with trace." << trace_id << " ------------------ ";
    return test_softmax(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst, psoft);
}