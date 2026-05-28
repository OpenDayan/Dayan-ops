#include "common.hpp"

#include <NPURuntime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __bin_kernels_pack;

static std::vector<int64_t> readInt64FromFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Open file failed: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % sizeof(int64_t) != 0) {
        throw std::runtime_error("Invalid trace file, file size cannot be divided by 8: " +
                                 filename);
    }

    std::vector<int64_t> buffer(size / sizeof(int64_t));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
        throw std::runtime_error("Error in reading file: " + filename);
    }
    return buffer;
}

template <typename T> static size_t bytes_of(const std::vector<T> &data) {
    return data.size() * sizeof(T);
}

static int tensor_bytes(const TensorParams &p) {
    if (p.len != 0) {
        return (int)p.len;
    }

    int elem_size = 4;
    if (p.type == "bf16") {
        elem_size = 2;
    } else if (p.type == "i64") {
        elem_size = 8;
    }

    int max_offset = 0;
    for (int i3 = 0; i3 < p.ne[3]; ++i3)
        for (int i2 = 0; i2 < p.ne[2]; ++i2)
            for (int i1 = 0; i1 < p.ne[1]; ++i1)
                for (int i0 = 0; i0 < p.ne[0]; ++i0) {
                    int off = i0 * p.nb[0] + i1 * p.nb[1] + i2 * p.nb[2] + i3 * p.nb[3];
                    if (off > max_offset) {
                        max_offset = off;
                    }
                }
    return max_offset + elem_size;
}

static void poison_dst_rows(std::vector<uint16_t> &dst, const std::vector<int64_t> &idx,
                            const TensorParams &psrc0, const TensorParams &psrc1,
                            const TensorParams &pdst) {
    const int nc = psrc0.ne[0];
    const int nr = psrc0.ne[1];
    const int ne02 = psrc0.ne[2];
    const int ne03 = psrc0.ne[3];
    const int ne11 = psrc1.ne[1];
    const int ne12 = psrc1.ne[2];

    for (int i03 = 0; i03 < ne03; ++i03) {
        for (int i02 = 0; i02 < ne02; ++i02) {
            const int i12 = i03 % ne12;
            const int i11 = i02 % ne11;
            for (int i = 0; i < nr; ++i) {
                const int64_t row = *(const int64_t *)((const char *)idx.data() + i * psrc1.nb[0] +
                                                       i11 * psrc1.nb[1] + i12 * psrc1.nb[2]);
                if (row < 0 || row >= pdst.ne[1]) {
                    continue;
                }

                for (int j = 0; j < nc; ++j) {
                    char *ptr = (char *)dst.data() + j * pdst.nb[0] + row * pdst.nb[1] +
                                i02 * pdst.nb[2] + i03 * pdst.nb[3];
                    *(uint16_t *)ptr = 0xffff;
                }
            }
        }
    }
}

static struct NPUKernelDesc *load_set_rows_desc(void) {
    static struct NPUKernelDesc *desc = NULL;
    if (desc != NULL) {
        return desc;
    }

    printf("[INFO] Initializing Runtime..\n");
    ssize_t err = npu_runtime_init(&runtime);
    if (err != NPU_SUCCESS) {
        printf("[ERROR] Failed to initialize Runtime: err=%ld\n", err);
        exit(-1);
    }
    std::cout << "Runtime initialized successfully." << std::endl;

    err = npu_runtime_load_kernelset(&runtime, &__bin_kernels_pack);
    if (err != NPU_SUCCESS) {
        printf("[ERROR] Failed to load kernel set: err=%ld\n", err);
        npu_runtime_deinit(&runtime);
        exit(-1);
    }
    printf("[INFO] Kernel set loaded successfully.\n");

    err = npu_runtime_get_kernel_desc(&runtime, "set_rows_f32_i64_bf16", &desc);
    if (err != NPU_SUCCESS) {
        printf("[ERROR] Failed to get kernel description set_rows_f32_i64_bf16: err=%ld\n", err);
        npu_runtime_deinit(&runtime);
        exit(-1);
    }
    printf("[INFO] Loaded set_rows_f32_i64_bf16 kernel desc successfully.\n");

    return desc;
}

static int host_set_rows_f32_i64_bf16(std::vector<float> &src0, std::vector<int64_t> &src1,
                                      std::vector<uint16_t> &dst, TensorParams &psrc0,
                                      TensorParams &psrc1, TensorParams &pdst) {
    struct NPUKernelDesc *desc = load_set_rows_desc();

    const uint64_t src0_dev = 0x81000000;
    const uint64_t src1_dev = 0x83000000;
    const uint64_t dst_dev = 0x85000000;

    ssize_t ret = npu_memcpy_to_dev(&runtime.dev_state, src0_dev, src0.data(), bytes_of(src0));
    if (ret < 0) {
        printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n", src0_dev, ret);
        return (int)ret;
    }

    ret = npu_memcpy_to_dev(&runtime.dev_state, src1_dev, src1.data(), bytes_of(src1));
    if (ret < 0) {
        printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n", src1_dev, ret);
        return (int)ret;
    }

    ret = npu_memcpy_to_dev(&runtime.dev_state, dst_dev, dst.data(), bytes_of(dst));
    if (ret < 0) {
        printf("[ERROR] Failed to write dst to device at 0x%lx: err=%ld\n", dst_dev, ret);
        return (int)ret;
    }

    const int nc = psrc0.ne[0];
    const int nr = psrc0.ne[1];
    const int ne1 = pdst.ne[1];
    const int ne02 = psrc0.ne[2];
    const int ne03 = psrc0.ne[3];
    const int ne11 = psrc1.ne[1];
    const int ne12 = psrc1.ne[2];

    int kernel_ret = 0;
    npu_runtime_kernel_start(&runtime, desc, &kernel_ret, 19, src0_dev, src1_dev, dst_dev,
                             (uint64_t)nc, (uint64_t)nr, (uint64_t)ne1, (uint64_t)ne02,
                             (uint64_t)ne03, (uint64_t)ne11, (uint64_t)ne12, (uint64_t)psrc0.nb[1],
                             (uint64_t)psrc0.nb[2], (uint64_t)psrc0.nb[3], (uint64_t)psrc1.nb[0],
                             (uint64_t)psrc1.nb[1], (uint64_t)psrc1.nb[2], (uint64_t)pdst.nb[1],
                             (uint64_t)pdst.nb[2], (uint64_t)pdst.nb[3]);
    if (kernel_ret != 0) {
        printf("[ERROR] Kernel set_rows_f32_i64_bf16 returned %d\n", kernel_ret);
        return kernel_ret;
    }

    ret = npu_memcpy_from_dev(&runtime.dev_state, dst.data(), dst_dev, bytes_of(dst));
    if (ret < 0) {
        printf("[ERROR] Failed to read dst from device at 0x%lx: err=%ld\n", dst_dev, ret);
        return (int)ret;
    }

    return 0;
}

static int compare_bf16(const std::vector<uint16_t> &result, const std::vector<uint16_t> &ref,
                        const TensorParams &pdst) {
    for (int i3 = 0; i3 < pdst.ne[3]; ++i3)
        for (int i2 = 0; i2 < pdst.ne[2]; ++i2)
            for (int i1 = 0; i1 < pdst.ne[1]; ++i1)
                for (int i0 = 0; i0 < pdst.ne[0]; ++i0) {
                    size_t off =
                        (i0 * pdst.nb[0] + i1 * pdst.nb[1] + i2 * pdst.nb[2] + i3 * pdst.nb[3]) /
                        sizeof(uint16_t);
                    if (result[off] != ref[off]) {
                        std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                        std::cerr << "  - result[" << i3 << "][" << i2 << "][" << i1 << "][" << i0
                                  << "]: 0x" << std::hex << result[off] << std::dec << std::endl;
                        std::cerr << "  - ref[" << i3 << "][" << i2 << "][" << i1 << "][" << i0
                                  << "]: 0x" << std::hex << ref[off] << std::dec << std::endl;
                        return 1;
                    }
                }

    std::cout << " *** Compare PASS ***" << std::endl;
    return 0;
}

static int test_set_rows(std::string src0_trace, std::string src1_trace, std::string dst_trace,
                         TensorParams &psrc0, TensorParams &psrc1, TensorParams &pdst) {
    try {
        if (psrc0.type != "f32" || psrc1.type != "i64" || pdst.type != "bf16") {
            std::cerr << "[ERROR] this set_rows test only supports f32/i64/bf16" << std::endl;
            return 1;
        }

        std::vector<float> src0 = readF32FromFile(src0_trace);
        std::vector<int64_t> src1 = readInt64FromFile(src1_trace);
        std::vector<uint16_t> ref = readUInt16FromFile(dst_trace);
        std::vector<uint16_t> result = ref;

        poison_dst_rows(result, src1, psrc0, psrc1, pdst);

        int ret = host_set_rows_f32_i64_bf16(src0, src1, result, psrc0, psrc1, pdst);
        if (ret != 0) {
            return ret;
        }

        return compare_bf16(result, ref, pdst);
    } catch (const std::exception &e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./test <TRACE DIR> <TRACE ID>" << std::endl;
        return 0;
    }

    std::string trace_dir(argv[1]);
    int trace_id = std::stoi(argv[2]);

    TraceParamsParser parser;
    TensorParams psrc0{}, psrc1{}, pdst{};
    std::string params = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    std::string src1_trace = trace_dir + "/" + std::to_string(trace_id) + "-src1.trace";
    parser.parse(params, &pdst, &psrc0, &psrc1, NULL, NULL, NULL, NULL);

    if (psrc0.ne.size() != 4 || psrc1.ne.size() != 4 || pdst.ne.size() != 4 ||
        psrc0.nb.size() != 4 || psrc1.nb.size() != 4 || pdst.nb.size() != 4) {
        std::cerr << "[ERROR] set_rows expects 4D tensor metadata" << std::endl;
        return 1;
    }
    if (tensor_bytes(psrc0) <= 0 || tensor_bytes(psrc1) <= 0 || tensor_bytes(pdst) <= 0) {
        std::cerr << "[ERROR] invalid tensor byte size" << std::endl;
        return 1;
    }

    std::cout << "Testing Set_Rows with trace." << trace_id << " ------------------ ";
    return test_set_rows(src0_trace, src1_trace, dst_trace, psrc0, psrc1, pdst);
}
