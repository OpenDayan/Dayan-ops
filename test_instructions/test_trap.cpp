#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

int main(int argc, char** argv) {
  std::cout << "Initializing Runtime..." << std::endl;
  struct NPURuntimeState runtime = {0};
  ssize_t err;
  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to initialize Runtime: " << err << std::endl;
    return -1;
  }
  long base;
  if(argc >= 2) {
    base = std::atol(argv[1]);
  }

  std::cout << "Runtime initialized successfully." << std::endl;

  // Get L2 cache info
  // test_get_l2_info(&runtime.dev_state);

  // Perform operations with the device
  err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to load kernel set: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  std::cout << "Kernel set loaded successfully." << std::endl;

  test_dump_mem(&runtime.dev_state, 0x80020000, 0x20);  // profiler

  struct NPUKernelDesc *kernel_desc;
  err = npu_runtime_get_kernel_desc(&runtime, "kernel_trap_testsuit", &kernel_desc);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to get kernel description: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  int kernel_ret;

  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 0);
  printf("ret:%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to start kernel: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  test_dump_mem(&runtime.dev_state, 0x80020000, 0x20);  // profiler

  npu_runtime_deinit(&runtime);

  return 0;
}
