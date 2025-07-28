#include "common.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>

#include <NPURuntime.h>

void permute_f32_2_1(std::vector<float>& a, std::vector<float>& res, TensorParams& pa, TensorParams& pres, PermuteParams& ppermute) {
    int bs = pa.ne[3];
    int n = pa.ne[2];
    int h = pa.ne[1];
    int w = pa.ne[0];

    int row_size = w * sizeof(float);

    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < h; ++k) {
                // Copy a[i][j][k][:] to res[i][k][j][:]
                const char * a_ptr = (char *)a.data() + k*pa.nb[1] + j*pa.nb[2] + i*pa.nb[3];
                size_t id = (k*n + j) * row_size;
                std::memcpy((char *)res.data() + id, a_ptr, row_size);
            }
        }
    }
}

static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
int host_permute_f32_2_1(float* a, float* res, int bs, int h, int n, int m){
    static struct NPUKernelDesc* permute_f32_2_1 = NULL;
    if (permute_f32_2_1 == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "permute_f32_2_1", &permute_f32_2_1);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded permute_f32_2_1 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    // uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, bs * h * n * m * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res, bs * h * n * m * sizeof(float));   // res
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%ld\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, permute_f32_2_1, &kernel_ret, 6,
                            a_dev, res_dev, bs, h, n, m);
    npu_memcpy_from_dev(&runtime.dev_state, res, res_dev, bs * h * n * m * sizeof(float)); // res
    return kernel_ret;
}

void kernel_permute_f32_2_1(std::vector<float>& a, std::vector<float>& res, TensorParams& pa, TensorParams& pres, PermuteParams& ppermute) {
    int bs = pa.ne[3];
    int h = pa.ne[2];
    int n = pa.ne[1];
    int m = pa.ne[0];

    host_permute_f32_2_1(a.data(),res.data(),bs,h,n,m);
}

int test_permute_f32(std::string src0_trace, std::string dst_trace, TensorParams& psrc0, TensorParams& pdst,
                     PermuteParams& ppermute) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);
        size_t len_src0 = psrc0.ne[0] * psrc0.ne[1] * psrc0.ne[2] * psrc0.ne[3];

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        std::vector<float> result_data;
        result_data.resize(len_src0);

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // kernel_permute_f32_2_1(data_a, result_data, ...);
        kernel_permute_f32_2_1(data_a, result_data, psrc0, pdst, ppermute);
        // --------------------------------------------
        // FIXME: CPU reference
        // permute_f32_2_1(data_a, result_data, psrc0, pdst, ppermute);


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

    std::string trace_dir(argv[1]);
    int trace_id = std::stoi(argv[2]);

    TraceParamsParser parser;
    TensorParams psrc0, pdst;
    PermuteParams ppermute;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    parser.parse(a, &pdst, &psrc0, NULL, NULL, NULL, NULL, &ppermute);
    if (psrc0.type != "f32" || !psrc0.is_contiguous) {
        std::cerr << "[ERROR] Permute tests only support permute from contiguous FP32" << std::endl;
        return 1;
    }
    if (ppermute.axis0 != 0 || ppermute.axis3 != 3 || ppermute.axis1 == 1 || ppermute.axis2 == 2) {
        std::cerr << "[ERROR] Permute tests only support (b, n, h, w) -> (b, h, n, w) permute" << std::endl;
        return 1;
    }
    std::cout << "Testing Permute with trace." << trace_id << " ------------------ ";

    // TODO: is_transpose
    return test_permute_f32(src0_trace, dst_trace, psrc0, pdst, ppermute);
}