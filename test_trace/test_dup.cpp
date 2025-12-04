#include "common.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <NPURuntime.h>

uint16_t float_to_fp16(float value) {
    uint32_t fbits = 0;
    std::memcpy(&fbits, &value, sizeof(float));
    unsigned int sign = (fbits >> 16) & 0x8000;
    fbits &= 0x7FFFFFFF;
    if (fbits >= 0x7F800000)
        return sign | 0x7C00 | ((fbits > 0x7F800000) ? (0x200 | ((fbits >> 13) & 0x3FF)) : 0);
    if (fbits >= 0x47800000)
        return sign | 0x7c00;
    if (fbits >= 0x38800000) {
        uint32_t value = sign | (((fbits >> 23) - 112) << 10) | ((fbits >> 13) & 0x3FF);
        uint32_t guard = (fbits >> 12) & 1;
        uint32_t sticky = (fbits & 0xFFF) != 0;
        return value + (guard & (sticky | value));
    }
    if (fbits >= 0x33000000) {
        int i = 125 - (fbits >> 23);
        fbits = (fbits & 0x7FFFFF) | 0x800000;
        uint32_t value = sign | (fbits >> (i + 1));
        uint32_t guard = (fbits >> i) & 1;
        uint32_t sticky = (fbits & ((static_cast<uint32_t>(1) << i) - 1)) != 0;
        return value + (guard & (sticky | value));
    }
    if (fbits != 0)
        return sign;
    return sign;
}

uint16_t f32_to_bf16(float h) {
    uint16_t res;
    union {
        float value;
        uint32_t bits;
    } fp32;
    fp32.value = h;
    if ((fp32.bits & 0x7fffffff) > 0x7f800000) { /* nan */
        res = (fp32.bits >> 16) | 64;            /* force to quiet */
        return res;
    }
    res = (fp32.bits + (0x7fff + ((fp32.bits >> 16) & 1))) >> 16;
    return res;
}

// contiguous f32 copy to contiguous bf16 (in prefill/decode stage)
void dup_cont_f32_cont_bf16(std::vector<float> &a, std::vector<uint16_t> &res, TensorParams &pa,
                            TensorParams &pres) {
    // if((pa.ne[3] != pres.ne[3]) ||
    //    (pa.ne[3] != pres.ne[3]) ||
    //    (pa.ne[3] != pres.ne[3]) ||
    //    (pa.ne[3] != pres.ne[3])) {
    //     std::cerr << "[ERROR] Contiguous fp32 and bf16 tensor has different shape" <<
    //     std::endl; return;
    // }
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Contiguous fp32 and bf16 tensor has different data num" << std::endl;
        return;
    }
    int bs = pa.ne[3];
    int h = pa.ne[2];
    int n = pa.ne[1];
    int m = pa.ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    res[i * h * n * m + j * n * m + k * m + p] =
                        f32_to_bf16(a[i * h * n * m + j * n * m + k * m + p]);
                }
            }
        }
    }
}

// non-contiguous f32 copy to non-contiguous bf16 (in decode stage)
void dup_cont_f32_no_cont_bf16(std::vector<float> &a, std::vector<uint16_t> &res, TensorParams &pa,
                               TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Contiguous fp32 and non-contiguous bf16 tensor has different "
                     "data num"
                  << std::endl;
        return;
    }
    int bs = pa.ne[3];
    int h = pa.ne[2];
    int n = pa.ne[1];
    int m = pa.ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    res[i * pres.nb[3] / sizeof(uint16_t) + j * pres.nb[2] / sizeof(uint16_t) +
                        k * pres.nb[1] / sizeof(uint16_t) + p * pres.nb[0] / sizeof(uint16_t)] =
                        f32_to_bf16(a[i * h * n * m + j * n * m + k * m + p]);
                }
            }
        }
    }
}

// contiguous f32 copy to non-contiguous bf16 (in prefill/decode stage)
void dup_no_cont_f32_no_cont_bf16(std::vector<float> &a, std::vector<uint16_t> &res,
                                  TensorParams &pa, TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Non-contiguous fp32 and non-contiguous tensor has different "
                     "data num"
                  << std::endl;
        return;
    }
    int bs = pa.ne[3];
    int h = pa.ne[2];
    int n = pa.ne[1];
    int m = pa.ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    res[i * pres.nb[3] / sizeof(uint16_t) + j * pres.nb[2] / sizeof(uint16_t) +
                        k * pres.nb[1] / sizeof(uint16_t) + p * pres.nb[0] / sizeof(uint16_t)] =
                        f32_to_bf16(a[i * pa.nb[3] / sizeof(float) + j * pa.nb[2] / sizeof(float) +
                                      k * pa.nb[1] / sizeof(float) + p * pa.nb[0] / sizeof(float)]);
                }
            }
        }
    }
}

// contiguous f32 copy to contiguous f32 (in decode stage)
void dup_cont_f32_cont_f32(std::vector<float> &a, std::vector<float> &res, TensorParams &pa,
                           TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Contiguous fp32 and fp32 tensor has different data num" << std::endl;
        return;
    }
    int bs = pa.ne[3];
    int h = pa.ne[2];
    int n = pa.ne[1];
    int m = pa.ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    res[i * h * n * m + j * n * m + k * m + p] =
                        a[i * h * n * m + j * n * m + k * m + p];
                }
            }
        }
    }
}
static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;
int host_dup_cont_f32_cont_f32(float *a, float *res, int a_bs, int a_h, int a_n, int a_m,
                               int res_bs, int res_h, int res_n, int res_m) {
    static struct NPUKernelDesc *dup_cont_f32_cont_f32 = NULL;
    if (dup_cont_f32_cont_f32 == NULL) {
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
        err =
            npu_runtime_get_kernel_desc(&runtime, "dup_cont_f32_cont_f32", &dup_cont_f32_cont_f32);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded dup_cont_f32_cont_f32 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    // uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a,
                            a_bs * a_h * a_n * a_m * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res,
                            res_bs * res_h * res_n * res_m * sizeof(float)); // res
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%ld\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, dup_cont_f32_cont_f32, &kernel_ret, 10, a_dev, res_dev, a_bs,
                             a_h, a_n, a_m, res_bs, res_h, res_n, res_m);
    npu_memcpy_from_dev(&runtime.dev_state, res, res_dev,
                        res_bs * res_h * res_n * res_m * sizeof(float)); // res
    return kernel_ret;
}

int host_dup_cont_f32_cont_bf16(float *a, uint16_t *res, int a_bs, int a_h, int a_n, int a_m,
                                int res_bs, int res_h, int res_n, int res_m) {
    static struct NPUKernelDesc *dup_cont_f32_cont_bf16 = NULL;
    if (dup_cont_f32_cont_bf16 == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "dup_cont_f32_cont_bf16",
                                          &dup_cont_f32_cont_bf16);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded dup_cont_f32_cont_bf16 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    // uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a,
                            a_bs * a_h * a_n * a_m * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res,
                            res_bs * res_h * res_n * res_m * sizeof(uint16_t)); // res
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%ld\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, dup_cont_f32_cont_bf16, &kernel_ret, 10, a_dev, res_dev,
                             a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m);
    npu_memcpy_from_dev(&runtime.dev_state, res, res_dev,
                        res_bs * res_h * res_n * res_m * sizeof(uint16_t)); // res
    return kernel_ret;
}

template <typename T>
int copy_to_dev_with_stride(uint64_t dev_base_addr, void *host_src, int bs, int h, int n, int m,
                            int nb3, int nb2, int nb1, int nb0) {
    for (int i3 = 0; i3 < bs; ++i3) {
        for (int i2 = 0; i2 < h; ++i2) {
            for (int i1 = 0; i1 < n; ++i1) {
                for (int i0 = 0; i0 < m; ++i0) {
                    uint64_t dst_offset = i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
                    // size_t src_flat_idx = (((i3 * h + i2) * n + i1) * m) +
                    // i0;
                    char *src_ptr = (char *)host_src + dst_offset;
                    ssize_t ret = npu_memcpy_to_dev(&runtime.dev_state, dev_base_addr + dst_offset,
                                                    (void *)src_ptr, sizeof(T));
                    if (ret < 0) {
                        printf("[ERROR] Failed to write element at "
                               "(%d,%d,%d,%d), offset 0x%lx\n",
                               i3, i2, i1, i0, dev_base_addr + dst_offset);
                        return ret;
                    }
                }
            }
        }
    }
    return 0;
}
template <typename T>
int copy_from_dev_with_stride(uint64_t dev_base_addr, void *host_dst, int bs, int h, int n, int m,
                              int nb3, int nb2, int nb1, int nb0) {
    for (int i3 = 0; i3 < bs; ++i3) {
        for (int i2 = 0; i2 < h; ++i2) {
            for (int i1 = 0; i1 < n; ++i1) {
                for (int i0 = 0; i0 < m; ++i0) {
                    uint64_t src_offset = i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;
                    char *dst_ptr = (char *)host_dst + src_offset;
                    ssize_t ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)dst_ptr,
                                                      dev_base_addr + src_offset, sizeof(T));
                    if (ret < 0) {
                        printf("[ERROR] Failed to read element at "
                               "(%d,%d,%d,%d), offset 0x%lx\n",
                               i3, i2, i1, i0, dev_base_addr + src_offset);
                        return ret;
                    }
                }
            }
        }
    }
    return 0;
}

int host_dup_cont_f32_no_cont_bf16(float *a, uint16_t *res, int a_bs, int a_h, int a_n, int a_m,
                                   int res_bs, int res_h, int res_n, int res_m, int a_nb3,
                                   int a_nb2, int a_nb1, int a_nb0, int res_nb3, int res_nb2,
                                   int res_nb1, int res_nb0) {
    static struct NPUKernelDesc *dup_cont_f32_no_cont_bf16 = NULL;
    if (dup_cont_f32_no_cont_bf16 == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "dup_cont_f32_no_cont_bf16",
                                          &dup_cont_f32_no_cont_bf16);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded dup_cont_f32_no_cont_bf16 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    // uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a,
                            a_bs * a_h * a_n * a_m * sizeof(float)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    // ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res, res_bs * res_h * res_n * res_m
    // * sizeof(uint16_t));   // res
    ret = copy_to_dev_with_stride<uint16_t>(res_dev, res, res_bs, res_h, res_n, res_m, res_nb3,
                                            res_nb2, res_nb1, res_nb0);
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%ld\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, dup_cont_f32_no_cont_bf16, &kernel_ret, 18, a_dev, res_dev,
                             a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m, a_nb3, a_nb2, a_nb1,
                             a_nb0, res_nb3, res_nb2, res_nb1, res_nb0);
    // npu_memcpy_from_dev(&runtime.dev_state, res, res_dev, res_bs * res_h * res_n * res_m *
    // sizeof(uint16_t)); // res
    copy_from_dev_with_stride<uint16_t>(res_dev, res, res_bs, res_h, res_n, res_m, res_nb3, res_nb2,
                                        res_nb1, res_nb0);
    return kernel_ret;
}
int host_dup_no_cont_f32_no_cont_bf16(float *a, uint16_t *res, int a_bs, int a_h, int a_n, int a_m,
                                      int res_bs, int res_h, int res_n, int res_m, int a_nb3,
                                      int a_nb2, int a_nb1, int a_nb0, int res_nb3, int res_nb2,
                                      int res_nb1, int res_nb0) {
    static struct NPUKernelDesc *dup_no_cont_f32_no_cont_bf16 = NULL;
    if (dup_no_cont_f32_no_cont_bf16 == NULL) {
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
        err = npu_runtime_get_kernel_desc(&runtime, "dup_no_cont_f32_no_cont_bf16",
                                          &dup_no_cont_f32_no_cont_bf16);
        if (err != NPU_SUCCESS) {
            printf("Failed to get kernel description: err=%ld\n", err);
            npu_runtime_deinit(&runtime);
            exit(-1);
        }
        printf("[INFO] Loaded dup_no_cont_f32_no_cont_bf16 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    // uint64_t b_dev = 0x83000000;
    uint64_t res_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    // ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, a_bs * a_h * a_n * a_m *
    // sizeof(float)); // a
    ret = copy_to_dev_with_stride<float>(a_dev, a, a_bs, a_h, a_n, a_m, a_nb3, a_nb2, a_nb1, a_nb0);
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%ld\n", a_dev, ret);
    // ret = npu_memcpy_to_dev(&runtime.dev_state, res_dev, res, res_bs * res_h * res_n * res_m
    // * sizeof(uint16_t));   // res
    ret = copy_to_dev_with_stride<uint16_t>(res_dev, res, res_bs, res_h, res_n, res_m, res_nb3,
                                            res_nb2, res_nb1, res_nb0);
    if (ret < 0)
        printf("[ERROR] Failed to write res to device at 0x%lx: err=%ld\n", res_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, dup_no_cont_f32_no_cont_bf16, &kernel_ret, 18, a_dev,
                             res_dev, a_bs, a_h, a_n, a_m, res_bs, res_h, res_n, res_m, a_nb3,
                             a_nb2, a_nb1, a_nb0, res_nb3, res_nb2, res_nb1, res_nb0);
    // npu_memcpy_from_dev(&runtime.dev_state, res, res_dev, res_bs * res_h * res_n * res_m *
    // sizeof(uint16_t)); // res
    copy_from_dev_with_stride<uint16_t>(res_dev, res, res_bs, res_h, res_n, res_m, res_nb3, res_nb2,
                                        res_nb1, res_nb0);
    return kernel_ret;
}

void kernel_dup_cont_f32_cont_f32(std::vector<float> &a, std::vector<float> &res, TensorParams &pa,
                                  TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Contiguous fp32 and fp32 tensor has different data num" << std::endl;
        return;
    }
    host_dup_cont_f32_cont_f32(a.data(), res.data(), pa.ne[3], pa.ne[2], pa.ne[1], pa.ne[0],
                               pres.ne[3], pres.ne[2], pres.ne[1], pres.ne[0]);
}
void kernel_dup_cont_f32_cont_bf16(std::vector<float> &a, std::vector<uint16_t> &res,
                                   TensorParams &pa, TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Contiguous fp32 and bf16 tensor has different data num" << std::endl;
        return;
    }
    host_dup_cont_f32_cont_bf16(a.data(), res.data(), pa.ne[3], pa.ne[2], pa.ne[1], pa.ne[0],
                                pres.ne[3], pres.ne[2], pres.ne[1], pres.ne[0]);
}
void kernel_dup_cont_f32_no_cont_bf16(std::vector<float> &a, std::vector<uint16_t> &res,
                                      TensorParams &pa, TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Contiguous fp32 and non-contiguous bf16 tensor has different "
                     "data num"
                  << std::endl;
        return;
    }
    host_dup_cont_f32_no_cont_bf16(a.data(), res.data(), pa.ne[3], pa.ne[2], pa.ne[1], pa.ne[0],
                                   pres.ne[3], pres.ne[2], pres.ne[1], pres.ne[0], pa.nb[3],
                                   pa.nb[2], pa.nb[1], pa.nb[0], pres.nb[3], pres.nb[2], pres.nb[1],
                                   pres.nb[0]);
}
void kernel_dup_no_cont_f32_no_cont_bf16(std::vector<float> &a, std::vector<uint16_t> &res,
                                         TensorParams &pa, TensorParams &pres) {
    if ((pa.ne[3] * pa.ne[2] * pa.ne[1] * pa.ne[0]) !=
        (pres.ne[3] * pres.ne[2] * pres.ne[1] * pres.ne[0])) {
        std::cerr << "[ERROR] Non-contiguous fp32 and non-contiguous bf16 tensor has "
                     "different data num"
                  << std::endl;
        return;
    }
    host_dup_no_cont_f32_no_cont_bf16(a.data(), res.data(), pa.ne[3], pa.ne[2], pa.ne[1], pa.ne[0],
                                      pres.ne[3], pres.ne[2], pres.ne[1], pres.ne[0], pa.nb[3],
                                      pa.nb[2], pa.nb[1], pa.nb[0], pres.nb[3], pres.nb[2],
                                      pres.nb[1], pres.nb[0]);
}

int test_dup_f32_to_bf16(std::string src0_trace, std::string dst_trace, TensorParams &psrc0,
                         TensorParams &pdst) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<uint16_t> data_ref = readUInt16FromFile(dst_trace);

        size_t len_src0 = psrc0.ne[0] * psrc0.ne[1] * psrc0.ne[2] * psrc0.ne[3];
        size_t len_dst = pdst.ne[0] * pdst.ne[1] * pdst.ne[2] * pdst.ne[3];
        if (len_src0 != len_dst) {
            std::cerr << "Error: Input " << src0_trace << " has different shape with " << dst_trace
                      << std::endl;
            return 1;
        }

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        std::vector<uint16_t> result_data;
        result_data.resize(data_ref.size());

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // if (pdst.is_contiguous)
        //     kernel_dup_cont_bf16(data_a, result_data, ...);
        // else
        //     kernel_dup_no_cont_bf16(data_a, result_data, ...);
        if (pdst.is_contiguous && psrc0.is_contiguous)
            kernel_dup_cont_f32_cont_bf16(data_a, result_data, psrc0, pdst);
        else if (!pdst.is_contiguous && !psrc0.is_contiguous)
            kernel_dup_no_cont_f32_no_cont_bf16(data_a, result_data, psrc0, pdst);
        else if (psrc0.is_contiguous && !pdst.is_contiguous)
            kernel_dup_cont_f32_no_cont_bf16(data_a, result_data, psrc0, pdst);
        else {
            std::cerr << "Not supported src0.is_contiguous=" << psrc0.is_contiguous
                      << ", dst.is_contiguous=" << pdst.is_contiguous << std::endl;
            return 1;
        }
        // --------------------------------------------
        //
        // FIXME: CPU reference
        // if (pdst.is_contiguous && psrc0.is_contiguous)
        //     dup_cont_f32_cont_bf16(data_a, result_data, psrc0, pdst);
        // else if (!pdst.is_contiguous && !psrc0.is_contiguous)
        //     dup_no_cont_f32_no_cont_bf16(data_a, result_data, psrc0, pdst);
        // else if (psrc0.is_contiguous && !pdst.is_contiguous)
        //     dup_cont_f32_no_cont_bf16(data_a, result_data, psrc0, pdst);
        // else {
        //     std::cerr << "Not supported src0.is_contiguous=" << psrc0.is_contiguous <<
        //                  ", dst.is_contiguous=" << pdst.is_contiguous << std::endl;
        //     return 1;
        // }

        // std::cout << "\nCompare with ref..." << std::endl;
        bool all_match = true;

        int bs = pdst.ne[3];
        int h = pdst.ne[2];
        int n = pdst.ne[1];
        int m = pdst.ne[0];
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < n; ++k) {
                    for (int p = 0; p < m; ++p) {
                        uint16_t ref = data_ref[i * pdst.nb[3] / sizeof(uint16_t) +
                                                j * pdst.nb[2] / sizeof(uint16_t) +
                                                k * pdst.nb[1] / sizeof(uint16_t) +
                                                p * pdst.nb[0] / sizeof(uint16_t)];
                        uint16_t result = result_data[i * pdst.nb[3] / sizeof(uint16_t) +
                                                      j * pdst.nb[2] / sizeof(uint16_t) +
                                                      k * pdst.nb[1] / sizeof(uint16_t) +
                                                      p * pdst.nb[0] / sizeof(uint16_t)];
                        if (ref != result) {
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                            std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << std::hex << result << std::dec << std::endl;
                            std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << std::hex << ref << std::dec << std::endl;
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

int test_dup_f32_to_f32(std::string src0_trace, std::string dst_trace, TensorParams &psrc0,
                        TensorParams &pdst) {
    try {
        std::vector<float> data_a = readF32FromFile(src0_trace);
        std::vector<float> data_ref = readF32FromFile(dst_trace);

        size_t len_src0 = psrc0.ne[0] * psrc0.ne[1] * psrc0.ne[2] * psrc0.ne[3];
        size_t len_dst = pdst.ne[0] * pdst.ne[1] * pdst.ne[2] * pdst.ne[3];
        if (len_src0 != len_dst) {
            std::cerr << "Error: Input " << src0_trace << " has different shape with " << dst_trace
                      << std::endl;
            return 1;
        }

        if (data_a.empty()) {
            std::cout << "[WARNING] src0 data is empty!" << std::endl;
            return 0;
        }

        std::vector<float> result_data;
        result_data.resize(data_ref.size());

        // FIXME: RUN KERNEL HERE
        //
        // --------------------------------------------
        // kernel_dup_no_cont_f32_cont_f32(data_a, result_data, ...);
        kernel_dup_cont_f32_cont_f32(data_a, result_data, psrc0, pdst);
        // --------------------------------------------
        //
        // FIXME: CPU reference
        // dup_cont_f32_cont_f32(data_a, result_data, psrc0, pdst);

        // std::cout << "\nCompare with ref..." << std::endl;
        bool all_match = true;

        int bs = pdst.ne[3];
        int h = pdst.ne[2];
        int n = pdst.ne[1];
        int m = pdst.ne[0];
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < n; ++k) {
                    for (int p = 0; p < m; ++p) {
                        float ref = data_ref[i * h * n * m + j * n * m + k * m + p];
                        float result = result_data[i * h * n * m + j * n * m + k * m + p];
                        if (ref != result) {
                            std::cerr << "\n !!! Compare FAILED !!!" << std::endl;
                            std::cerr << "  - result[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << std::hex << result << std::dec << std::endl;
                            std::cerr << "  - ref[" << i << "][" << j << "][" << k << "][" << p
                                      << "]:    " << std::hex << ref << std::dec << std::endl;
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
    TensorParams psrc0, pdst;
    std::string a = trace_dir + "/" + std::to_string(trace_id) + "-params.txt";
    std::string dst_trace = trace_dir + "/" + std::to_string(trace_id) + "-dst.trace";
    std::string src0_trace = trace_dir + "/" + std::to_string(trace_id) + "-src0.trace";
    parser.parse(a, &pdst, &psrc0, NULL, NULL, NULL, NULL, NULL);
    if (!(psrc0.type == "f32" && pdst.type == "f32") &&
        !(psrc0.type == "f32" && pdst.type == "bf16")) {
        std::cerr << "[ERROR] Dup tests only support FP32 to BF16/Fp32 test" << std::endl;
        return 1;
    }
    // dup f32 to bf16
    if (psrc0.type == "f32" && pdst.type == "bf16" && !psrc0.is_contiguous && pdst.is_contiguous) {
        std::cerr << "[ERROR] Dup tests does not support non-contiguous src0 and "
                     "contiguous dst for FP32 to BF16 test"
                  << std::endl;
        return 1;
    }
    // dup f32 to f32 输出张量连续
    if (psrc0.type == "f32" && pdst.type == "f32" &&
        (!pdst.is_contiguous || !psrc0.is_contiguous)) {
        std::cerr << "[ERROR] Dup tests only support contiguous FP32 to FP32 test" << std::endl;
        return 1;
    }
    std::cout << "Testing Dup with trace." << trace_id << " ------------------ ";
    if (pdst.type == "bf16")
        return test_dup_f32_to_bf16(src0_trace, dst_trace, psrc0, pdst);
    return test_dup_f32_to_f32(src0_trace, dst_trace, psrc0, pdst);
}