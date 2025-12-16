#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"

extern struct NPUDeviceKernelSetDesc __bin_kernels_pack;

static inline long run_instr_test(struct NPURuntimeState *runtime, std::string test_name) {

    std::cout << "Running " << test_name << "..." << std::endl;

    struct NPUKernelDesc *kernel_desc;
    ssize_t err = npu_runtime_get_kernel_desc(runtime, test_name.c_str(), &kernel_desc);
    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to get kernel description: " << test_name << std::endl;
        return -1;
    }

    int kernel_ret;
    err = npu_runtime_kernel_start(runtime, kernel_desc, &kernel_ret, 0);

    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to start kernel: " << test_name << std::endl;
        return -1;
    }

    if (kernel_ret == 0) {
        std::cout << "Test " << test_name << " passed!" << std::endl;
    } else {
        std::cout << "Test " << test_name << " failed!" << std::endl;
    }
    return kernel_ret;
}

int main(int argc, char **argv) {
    std::cout << "Initializing Runtime..." << std::endl;
    struct NPURuntimeState runtime = {0};
    ssize_t err;
    err = npu_runtime_init(&runtime);
    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to initialize Runtime: " << err << std::endl;
        return -1;
    }
    std::cout << "Runtime initialized successfully." << std::endl;

    err = npu_runtime_load_kernelset(&runtime, &__bin_kernels_pack);
    if (err != NPU_SUCCESS) {
        std::cerr << "Failed to load kernel set: " << err << std::endl;
        npu_runtime_deinit(&runtime);
        return -1;
    }

    std::cout << "Kernel set loaded successfully." << std::endl;

    // run_instr_test(&runtime, "test_vec_tcsr_write_read");
    // run_instr_test(&runtime, "test_vec_base_test");
    // run_instr_test(&runtime, "test_lsu_tcsr_write_read");
    // run_instr_test(&runtime, "test_lsu_base_transpose_test");
    // run_instr_test(&runtime, "test_mat_tcsr_write_read");
    // run_instr_test(&runtime, "test_calculated_goto");
    // run_instr_test(&runtime, "test_npu_task_basic_dma");
    run_instr_test(&runtime, "test_legacy_add_int32_inst");

    std::cout << "All Instr Test Finished." << std::endl;

    npu_runtime_deinit(&runtime);

    return 0;
}
