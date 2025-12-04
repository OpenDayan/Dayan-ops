#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NPURuntime.h>

#include "common.hpp"

static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

void add(std::vector<float> &a, std::vector<float> &b, std::vector<float> &res, TensorParams &psrc0,
         TensorParams &psrc1) {
    int bs = psrc0.ne[3];
    int h = psrc0.ne[2];
    int n = psrc0.ne[1];
    int m = psrc0.ne[0];
    for (int i = 0; i < bs; ++i) {
        int i1 = psrc1.ne[3] == 1 ? 0 : i;
        for (int j = 0; j < h; ++j) {
            int j1 = psrc1.ne[2] == 1 ? 0 : j;
            for (int k = 0; k < n; ++k) {
                int k1 = psrc1.ne[1] == 1 ? 0 : k;
                for (int p = 0; p < m; ++p) {
                    int p1 = psrc1.ne[0] == 1 ? 0 : p;
                    res[i * h * n * m + j * n * m + k * m + p] =
                        a[i * h * n * m + j * n * m + k * m + p] +
                        b[i1 * h * n * m + j1 * n * m + k1 * m + p1];
                }
            }
        }
    }
}

int host_kernel_add_fp32(void *a, void *b, void *c, int total_elements, const TensorParams &psrc0,
                         const TensorParams &psrc1) {
    printf("\nTotal elements: %d\n", total_elements);
    printf("psrc0 shape: [%d, %d, %d, %d]\n", psrc0.ne[3], psrc0.ne[2], psrc0.ne[1], psrc0.ne[0]);
    printf("psrc1 shape: [%d, %d, %d, %d]\n", psrc1.ne[3], psrc1.ne[2], psrc1.ne[1], psrc1.ne[0]);

    static struct NPUKernelDesc *kernel_add_fp32_desc = NULL;
    if (kernel_add_fp32_desc == NULL) {
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
            printf("Failed to load kernel set: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Kernel set loaded successfully.\n");
        err = npu_runtime_get_kernel_desc(&runtime, "kernel_add_fp32", &kernel_add_fp32_desc);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded kernel_add_fp32 kernel desc successfully.\n");
    }

    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;

    // Calculate memory size needed for tensor B (considering broadcast)
    int b_total_elements = psrc1.ne[3] * psrc1.ne[2] * psrc1.ne[1] * psrc1.ne[0];

    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, total_elements * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, b_total_elements * sizeof(float)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%ld\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, total_elements * sizeof(float)); // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%ld\n", c_dev, ret);

    int kernel_ret;

    npu_runtime_kernel_start(&runtime, kernel_add_fp32_desc, &kernel_ret, 11, a_dev, b_dev, c_dev,
                             psrc0.ne[0], psrc0.ne[1], psrc0.ne[2], psrc0.ne[3], psrc1.ne[0],
                             psrc1.ne[1], psrc1.ne[2], psrc1.ne[3]);

    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, total_elements * sizeof(float)); // c
    return kernel_ret;
}

void kernel_add(std::vector<float> &a, std::vector<float> &b, std::vector<float> &res,
                TensorParams &psrc0, TensorParams &psrc1) {
    int total_elements = psrc0.ne[3] * psrc0.ne[2] * psrc0.ne[1] * psrc0.ne[0];

    int kernel_result =
        host_kernel_add_fp32(a.data(), b.data(), res.data(), total_elements, psrc0, psrc1);

    assert(kernel_result == 0);
}

int test_add(std::string src0_trace, std::string src1_trace, std::string dst_trace,
             TensorParams &psrc0, TensorParams &psrc1) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<float> data_b = readF32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

        if (data_a.size() != data_ref.size()) {
            std::cerr << "Error: Input " << src0_trace << " has different size with " << dst_trace
                      << std::endl;
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
        kernel_add(data_a, data_b, result_data, psrc0, psrc1);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        // add(data_a, data_b, result_data, psrc0, psrc1);

        // 对比结果
        // std::cout << "\nCompare with ref..." << std::endl;
        const float epsilon = 1e-6f;
        bool all_match = true;

        int bs = psrc0.ne[3];
        int h = psrc0.ne[2];
        int n = psrc0.ne[1];
        int m = psrc0.ne[0];
        for (size_t i = 0; i < bs; ++i) {
            size_t i1 = psrc1.ne[3] == 1 ? 0 : i;
            for (size_t j = 0; j < h; ++j) {
                size_t j1 = psrc1.ne[2] == 1 ? 0 : j;
                for (size_t k = 0; k < n; ++k) {
                    size_t k1 = psrc1.ne[1] == 1 ? 0 : k;
                    for (size_t p = 0; p < m; ++p) {
                        size_t p1 = psrc1.ne[0] == 1 ? 0 : p;

                        float result = result_data[i * h * n * m + j * n * m + k * m + p];
                        float ref = data_ref[i * h * n * m + j * n * m + k * m + p];

                        if (std::abs(result - ref) > epsilon) {
                            std::cerr << std::fixed << std::setprecision(8);
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                            std::cerr
                                << "  - input0[" << i << "][" << j << "][" << k << "][" << p
                                << "]:    " << data_a[i1 * h * n * m + j1 * n * m + k1 * m + p1]
                                << std::endl;
                            std::cerr
                                << "  - input1[" << i << "][" << j << "][" << k << "][" << p
                                << "]:    " << data_b[i1 * h * n * m + j1 * n * m + k1 * m + p1]
                                << std::endl;
                            std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << result << std::endl;
                            std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << ref << std::endl;
                            std::cerr << "  - result - ref:        " << std::abs(result - ref)
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
    if (!pdst.is_contiguous || !psrc0.is_contiguous || !psrc1.is_contiguous) {
        std::cerr << "[ERROR] Add tests only support contiguous tensor now!" << std::endl;
        return 1;
    }
    std::cout << "Testing Add with trace." << trace_id << " ------------------ ";
    return test_add(src0_trace, src1_trace, dst_trace, psrc0, psrc1);
}