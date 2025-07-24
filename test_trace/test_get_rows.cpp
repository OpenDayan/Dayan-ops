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
#include <cassert>

#include "common.hpp"

void get_rows_bf16(std::vector<uint16_t>& a, std::vector<int>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
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
                int row_idx = b[j*n1*m1 + k*m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    std::cerr << "[ERROR] invalid row_idx value: " << row_idx << std::endl;
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    uint32_t val_u32 = (uint32_t)a[j*h0*n0*m0 + k*n0*m0 + row_idx*m0 + l];
                    val_u32 <<= 16;
                    float val_f32;
                    std::memcpy(&val_f32, &val_u32, sizeof(float));
                    res[j*n1*m1*m0 + k*m0*m1 + p*m0 + l] = val_f32;
                }

            }
}

void get_rows_f32(std::vector<float>& a, std::vector<int>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
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
                int row_idx = b[j*n1*m1 + k*m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    std::cerr << "[ERROR] invalid row_idx value: " << row_idx << std::endl;
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    res[j*n1*m1*m0 + k*m0*m1 + p*m0 + l] = a[j*h0*n0*m0 + k*n0*m0 + row_idx*m0 + l];
                }

            }
}

//testing kernel function
#include <NPURuntime.h>
static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
long host_get_rows(void *a, void *b, void *c, 
                        int h1, int n1, int m1, int m0, int n0,
                        bool isbf16){
    printf("\nh1:%d, n1:%d, m1:%d, m0:%d, n0:%d\n", h1, n1, m1, m0, n0);
    int src_elem_size = sizeof(float);
    if(isbf16){
        printf("Using bf16 type input, will convert to f32\n");
        src_elem_size = sizeof(uint16_t);
    }
    static struct NPUKernelDesc *get_rows_f32_desc = NULL;
    // if (get_rows_f32_desc == NULL)
    // {
    // Since two types may be used during testing, reinitialization is performed each time.
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
        if(isbf16) {
            printf("[INFO] Trying to load get_rows_bf16_i32_f32...\n");
            err = npu_runtime_get_kernel_desc(&runtime, "get_rows_bf16_i32_f32", &get_rows_f32_desc);
        }
        else{
            printf("[INFO] Trying to load get_rows_f32_i32_f32...\n");
            err = npu_runtime_get_kernel_desc(&runtime, "get_rows_f32_i32_f32", &get_rows_f32_desc);
        }
        if (err != NPU_SUCCESS) {
          printf("Failed to get kernel description: err=%ld\n", err);
          npu_runtime_deinit(&runtime);
          exit(-1);
        }
        printf("[INFO] Loaded get_rows kernel desc successfully.\n");
    // }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;
    // Copy data to device

    ssize_t ret = 0;
    // printf("[INFO] Ready to load data to dev\n");
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, h1 * n1 * n0 * m0 * src_elem_size); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    // printf("Wrote a to device at 0x%lx with ret:%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, 1 * h1 * n1 * m1 * sizeof(int)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%ld\n", b_dev, ret);
    // printf("Wrote b to device at 0x%lx with ret:%ld\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, h1 * n1 * m1 * m0 * sizeof(float));   // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%ld\n", c_dev, ret);
    // printf("Wrote c to device at 0x%lx with ret:%ld\n", c_dev, ret);
    int kernel_ret;
    // printf("[INFO] Ready to start get rows npu kernel\n");
    npu_runtime_kernel_start(&runtime, get_rows_f32_desc, &kernel_ret, 8,
                             a_dev, b_dev, c_dev,
                             (uint64_t)h1, (uint64_t)n1, (uint64_t)m1, (uint64_t)m0, (uint64_t)n0);
    // printf("[INFO] Finished get rows npu kernel, ready for copy from dev\n");
    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, h1 * n1 * m1 * m0 * sizeof(float)); // c
    return kernel_ret;
}

void kernel_get_rows_f32(std::vector<float>& a, std::vector<int>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst){
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

    host_get_rows(a.data(), b.data(), res.data(), h1, n1, m1, m0, n0, false);

}
void kernel_get_rows_bf16(std::vector<uint16_t>& a, std::vector<int>& b, std::vector<float>& res,
                   TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst){
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

    host_get_rows(a.data(), b.data(), res.data(), h1, n1, m1, m0, n0, true);

}
#include <random>
#include <cassert>
#include <chrono>
#include <vector>
#include <cstdio>

template<typename T>
void generate_random_test_data(
    std::vector<T>& a, std::vector<int>& b,
    TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint16_t>,
                  "Only float or uint16_t are supported for a");

    std::random_device rd;
    int seed = rd();
    printf("[INFO] Generating random test data using seed = %d\n", seed);
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> dim_dist(1, 20);
    std::uniform_int_distribution<int> x_dist(1, 20);

    int h1 = dim_dist(rng);
    int n1 = dim_dist(rng);
    int m0 = dim_dist(rng);
    int m1 = dim_dist(rng);
    int x  = x_dist(rng);

    psrc0.ne.resize(4);
    psrc1.ne.resize(4);
    pdst.ne.resize(4);

    psrc0.ne[0] = m0;
    psrc0.ne[1] = x;
    psrc0.ne[2] = n1;
    psrc0.ne[3] = h1;

    psrc1.ne[0] = m1;
    psrc1.ne[1] = n1;
    psrc1.ne[2] = h1;
    psrc1.ne[3] = 1;

    pdst.ne[0] = m0;
    pdst.ne[1] = m1;
    pdst.ne[2] = n1;
    pdst.ne[3] = h1;

    size_t a_size = h1 * n1 * x * m0;
    a.resize(a_size);

    if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
        for (auto& v : a)
            v = val_dist(rng);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
        for (auto& v : a) {
            float f = val_dist(rng);
            uint32_t tmp;
            std::memcpy(&tmp, &f, sizeof(float));
            v = static_cast<uint16_t>(tmp >> 16);  // high 16 bit as bf16
        }
    }

    size_t b_size = h1 * n1 * m1;
    b.resize(b_size);

    std::uniform_int_distribution<int> index_dist(0, x - 1);
    for (auto& idx : b)
        idx = index_dist(rng);  // make sure row index is legal

    printf("Generated test data:\n");
    printf("psrc0 shape: (%d, %d, %d, %d)\n", psrc0.ne[3], psrc0.ne[2], psrc0.ne[1], psrc0.ne[0]);
    printf("psrc1 shape: (%d, %d, %d, %d)\n", psrc1.ne[3], psrc1.ne[2], psrc1.ne[1], psrc1.ne[0]);
    printf("pdst  shape: (%d, %d, %d, %d)\n", pdst.ne[3], pdst.ne[2], pdst.ne[1], pdst.ne[0]);
}

template<typename T>
int test_random_case() {
    std::vector<T> a;
    std::vector<int> b;
    TensorParams psrc0, psrc1, pdst;

    generate_random_test_data<T>(a, b, psrc0, psrc1, pdst);

    size_t dst_size = pdst.ne[0] * pdst.ne[1] * pdst.ne[2] * pdst.ne[3];
    std::vector<float> data_ref(dst_size);
    std::vector<float> result_data(dst_size);

    if constexpr (std::is_same_v<T, uint16_t>) {
        get_rows_bf16(a, b, data_ref, psrc0, psrc1, pdst);
        kernel_get_rows_bf16(a, b, result_data, psrc0, psrc1, pdst);
    } else if constexpr (std::is_same_v<T, float>) {
        get_rows_f32(a, b, data_ref, psrc0, psrc1, pdst);
        kernel_get_rows_f32(a, b, result_data, psrc0, psrc1, pdst);
    } else {
        printf("Unsupported type\n");
    }

    const float epsilon = 1e-6f;
    bool all_match = true;

    int bs = pdst.ne[3];
    int h = pdst.ne[2];
    int n = pdst.ne[1];
    int m = pdst.ne[0];

    std::cout << "Testing random case...\n";
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    size_t idx = i * h * n * m + j * n * m + k * m + p;
                    assert(idx < result_data.size());
                    float result = static_cast<float>(result_data[idx]);
                    float ref = static_cast<float>(data_ref[idx]);
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
        } else {
            return 1;
        }
}

int test_get_rows(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                  TensorParams& psrc0, TensorParams& psrc1, TensorParams& pdst) {
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

    test_random_case<float>();
    test_random_case<uint16_t>();
    // printf("test_random_case() returned to main\n");
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