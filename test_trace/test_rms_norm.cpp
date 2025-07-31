#include <cstddef>
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


static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
int host_rms_norm_f32(
    void *x, void *y, int b, int h, int n, int m, float epsilon)
{
    printf("\nb:%d, h:%d, n:%d, m:%d\n", b, h, n, m);
    printf("epsilon: %f\n", epsilon);
    static struct NPUKernelDesc *rms_norm_f32_desc = NULL;
    if (rms_norm_f32_desc == NULL)
    {
        ssize_t err = 0;
        printf("[INFO] Initializing Runtime..\n");
        err = npu_runtime_init(&runtime);
        if (err != NPU_SUCCESS) {
            printf("[ERROR] Failed to initialize Runtime: err=%ld\n", err);
            exit(-1);
        }
        printf("[INFO] Runtime initialized successfully.\n");

        err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
        if (err != NPU_SUCCESS) {
            printf("[ERROR] Failed to load kernel set: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Kernel set loaded successfully.\n");

        err = npu_runtime_get_kernel_desc(&runtime, "rms_norm_f32", &rms_norm_f32_desc);
        if (err != NPU_SUCCESS) {
            printf("[ERROR] Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded rms_norm_f32 kernel desc successfully.\n");
    }

    // Static device addresses for input/output data
    uint64_t x_dev = 0x81000000;
    uint64_t y_dev = 0x83000000;

    // Copy data to device (x and y tensors)
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, x_dev, x, b * h * n * m * sizeof(float)); // x
    if (ret < 0)
        printf("[ERROR] Failed to write x to device at 0x%lx: err=%ld\n", x_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, y_dev, y, b * h * n * m * sizeof(float)); // y
    if (ret < 0)
        printf("[ERROR] Failed to write y to device at 0x%lx: err=%ld\n", y_dev, ret);
    // Launch kernel
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, rms_norm_f32_desc, &kernel_ret, 7,
                             x_dev, y_dev, (uint64_t)b, (uint64_t)h, (uint64_t)n, (uint64_t)m, (uint64_t)epsilon);

    // Copy result from device to host
    ret = npu_memcpy_from_dev(&runtime.dev_state, y, y_dev, b * h * n * m * sizeof(float)); // y
    if (ret < 0)
        printf("[ERROR] Failed to read y from device at 0x%lx: err=%ld\n", y_dev, ret);

    return kernel_ret;
}


void kernel_rms_norm(
    std::vector<float>& a, std::vector<float>& res,
    TensorParams& psrc0, TensorParams & pdst, NormParams & pnorm) {
    int b = psrc0.ne[3];
    int h = psrc0.ne[2];
    int n = psrc0.ne[1];
    int m = psrc0.ne[0];

    float epsilon = (float)pnorm.eps;
    
    // Call Kernel
    host_rms_norm_f32(
        a.data(), res.data(),
        b, h, n, m, epsilon);
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
        kernel_rms_norm(data_a, result_data, psrc0, pdst, pnorm);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        // rms_norm(data_a, result_data, psrc0, pnorm);

        // 对比结果
        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 1e-6f;
        bool all_match = true;

        for (size_t i = 0; i < result_data.size(); ++i) {
            if (std::abs(result_data[i] - data_ref[i]) > epsilon) {
                std::cerr << std::fixed << std::setprecision(8);
                std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                
                // Print first 9 mismatches
                size_t mismatches_to_print = std::min((size_t)6, result_data.size());
                std::cerr << "First " << mismatches_to_print << " mismatches:" << std::endl;
                
                for (size_t j = 0; j < mismatches_to_print; j++) {
                    std::cerr << "[" << j << "]:" << std::endl;
                    std::cerr << "  - input1[" << j / psrc0.ne[0] << "][" << j % pdst.ne[0] << "]:    " << data_a[j] << std::endl;
                    std::cerr << "  - result[" << j / pdst.ne[0] << "][" << j % pdst.ne[0] << "]:    " << result_data[j] << std::endl;
                    std::cerr << "  - ref[" << j / pdst.ne[0] << "][" << j % pdst.ne[0] << "]:    " << data_ref[j] << std::endl;
                    std::cerr << "  - result - ref:        " << std::abs(result_data[j] - data_ref[j]) << std::endl;
                    std::cerr << std::endl;
                }
                
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
    parser.parse(a, &pdst, &psrc0, NULL, &pnorm, NULL, NULL, NULL);
    if (!pdst.is_contiguous || !psrc0.is_contiguous) {
        std::cerr << "[ERROR] RMS_Norm tests only support contiguous tensor now!" << std::endl;
        return 1;
    }
    std::cout << "Testing RMS_Norm with trace." << trace_id << " ------------------ ";
    return test_rms_norm(src0_trace, dst_trace, psrc0, pdst, pnorm);
}