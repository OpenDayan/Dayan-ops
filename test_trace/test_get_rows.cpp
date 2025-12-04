#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

void get_rows_bf16(std::vector<uint16_t> &a, std::vector<int> &b, std::vector<float> &res,
                   TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    // source tensor
    int bs0 = psrc0.ne[3];
    int h0 = psrc0.ne[2];
    int n0 = psrc0.ne[1];
    int m0 = psrc0.ne[0];

    // index tensor
    int bs1 = psrc1.ne[3];
    int h1 = psrc1.ne[2];
    int n1 = psrc1.ne[1];
    int m1 = psrc1.ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

    for (int j = 0; j < h1; ++j)
        for (int k = 0; k < n1; ++k)
            for (int p = 0; p < m1; ++p) {
                // row_idx = b[0][j][k][p]
                int row_idx = b[j * n1 * m1 + k * m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    std::cerr << "[ERROR] invalid row_idx value: " << row_idx << std::endl;
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    uint32_t val_u32 =
                        (uint32_t)a[j * h0 * n0 * m0 + k * n0 * m0 + row_idx * m0 + l];
                    val_u32 <<= 16;
                    float val_f32;
                    std::memcpy(&val_f32, &val_u32, sizeof(float));
                    res[j * n1 * m1 * m0 + k * m0 * m1 + p * m0 + l] = val_f32;
                }
            }
}

void get_rows_f32(std::vector<float> &a, std::vector<int> &b, std::vector<float> &res,
                  TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    // source tensor
    int bs0 = psrc0.ne[3];
    int h0 = psrc0.ne[2];
    int n0 = psrc0.ne[1];
    int m0 = psrc0.ne[0];

    // index tensor
    int bs1 = psrc1.ne[3];
    int h1 = psrc1.ne[2];
    int n1 = psrc1.ne[1];
    int m1 = psrc1.ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    // assert ((bs0 == h1) && (h0 == n1))

    // sr0 shape: (h1, n1, x, m0)  (x is the number range of src0[0][j][k][p])
    // sr1 shape: (1, h1, n1, m1)
    // dst shape: (h1, n1, m1, m0)

    for (int j = 0; j < h1; ++j)
        for (int k = 0; k < n1; ++k)
            for (int p = 0; p < m1; ++p) {
                // row_idx = b[0][j][k][p]
                int row_idx = b[j * n1 * m1 + k * m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    std::cerr << "[ERROR] invalid row_idx value: " << row_idx << std::endl;
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    res[j * n1 * m1 * m0 + k * m0 * m1 + p * m0 + l] =
                        a[j * h0 * n0 * m0 + k * n0 * m0 + row_idx * m0 + l];
                }
            }
}

#include <NPURuntime.h>
static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
long host_get_rows_f32(void *a, void *b, void *c, int h1, int n1, int m1, int m0, int n0) {
    // printf("\nh1:%d, n1:%d, m1:%d, m0:%d, n0:%d\n", h1, n1, m1, m0, n0);
    static struct NPUKernelDesc *get_rows_f32_desc = NULL;
    if (get_rows_f32_desc == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "get_rows_f32_i32_f32", &get_rows_f32_desc);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded get_rows_f32_i32_f32 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;
    // Copy data to device

    ssize_t ret = 0;
    // printf("[INFO] Ready to load data to dev\n");
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, h1 * n1 * n0 * m0 * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, 1 * h1 * n1 * m1 * sizeof(int)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%ld\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, h1 * n1 * m1 * m0 * sizeof(float)); // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%ld\n", c_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, get_rows_f32_desc, &kernel_ret, 8, a_dev, b_dev, c_dev,
                             (uint64_t)h1, (uint64_t)n1, (uint64_t)m1, (uint64_t)m0, (uint64_t)n0);
    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, h1 * n1 * m1 * m0 * sizeof(float)); // c
    return kernel_ret;
}
long host_get_rows_bf16(void *a, void *b, void *c, int h1, int n1, int m1, int m0, int n0) {
    static struct NPUKernelDesc *get_rows_bf16_desc = NULL;
    if (get_rows_bf16_desc == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "get_rows_bf16_i32_f32", &get_rows_bf16_desc);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded get_rows_bf16_i32_f32 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a,
                            h1 * n1 * n0 * m0 * sizeof(uint16_t)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, 1 * h1 * n1 * m1 * sizeof(int)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%ld\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, h1 * n1 * m1 * m0 * sizeof(float)); // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%ld\n", c_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, get_rows_bf16_desc, &kernel_ret, 8, a_dev, b_dev, c_dev,
                             (uint64_t)h1, (uint64_t)n1, (uint64_t)m1, (uint64_t)m0, (uint64_t)n0);
    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, h1 * n1 * m1 * m0 * sizeof(float)); // c
    return kernel_ret;
}

void kernel_get_rows_f32(std::vector<float> &a, std::vector<int> &b, std::vector<float> &res,
                         TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    // source tensor
    int bs0 = psrc0.ne[3];
    int h0 = psrc0.ne[2];
    int n0 = psrc0.ne[1];
    int m0 = psrc0.ne[0];

    // index tensor
    int bs1 = psrc1.ne[3];
    int h1 = psrc1.ne[2];
    int n1 = psrc1.ne[1];
    int m1 = psrc1.ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    host_get_rows_f32(a.data(), b.data(), res.data(), h1, n1, m1, m0, n0);
}
void kernel_get_rows_bf16(std::vector<uint16_t> &a, std::vector<int> &b, std::vector<float> &res,
                          TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    // source tensor
    int bs0 = psrc0.ne[3];
    int h0 = psrc0.ne[2];
    int n0 = psrc0.ne[1];
    int m0 = psrc0.ne[0];

    // index tensor
    int bs1 = psrc1.ne[3];
    int h1 = psrc1.ne[2];
    int n1 = psrc1.ne[1];
    int m1 = psrc1.ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    host_get_rows_bf16(a.data(), b.data(), res.data(), h1, n1, m1, m0, n0);
}

int test_get_rows(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                  TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    try {

        std::vector<int> data_b = readInt32FromFile(src1_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);
        std::vector<float> result_data;

        if (psrc0.type == "bf16") {
            std::vector<uint16_t> data_a = readUInt16FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            result_data.resize(data_a.size());

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------------
            // kernel_get_rows_bf16(data_a, data_b, result_data, ...);
            kernel_get_rows_bf16(data_a, data_b, result_data, psrc0, psrc1, pdst);
            // --------------------------------------------
            //
            // FIXME: CPU reference
            // get_rows_bf16(data_a, data_b, result_data, psrc0, psrc1, pdst);

        } else if (psrc0.type == "f32") {
            std::vector<float> data_a = readF32FromFile(src0_trace);
            if (data_a.empty()) {
                std::cout << "[WARNING] src0 data is empty!" << std::endl;
                return 0;
            }

            result_data.resize(data_a.size());

            // FIXME: RUN KERNEL HERE
            //
            // --------------------------------------------
            // kernel_get_rows_f32(data_a, data_b, result_data, ...);
            kernel_get_rows_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);
            // --------------------------------------------
            //
            // FIXME: CPU reference
            // get_rows_f32(data_a, data_b, result_data, psrc0, psrc1, pdst);
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
                        float result = result_data[i * h * n * m + j * n * m + k * m + p];
                        float ref = data_ref[i * h * n * m + j * n * m + k * m + p];
                        if (std::abs(result - ref) > epsilon) {
                            std::cerr << std::fixed << std::setprecision(8);
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
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
        std::cerr << "[ERROR] get_rows tests only support contiguous tensor now!" << std::endl;
        return 1;
    }
    std::cout << "Testing Get_Rows with trace." << trace_id << " ------------------ ";
    return test_get_rows(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst);
}