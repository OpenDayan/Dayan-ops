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
  std::cout << "Runtime initialized successfully." << std::endl;

  err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to load kernel set: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  std::cout << "Kernel set loaded successfully." << std::endl;

  test_dump_mem(&runtime.dev_state, runtime.kernel_set_desc.perf, 0x10);  // profiler
  test_dump_mem(&runtime.dev_state, 0x81000000, 0x100); // args tester

  struct NPUKernelDesc *kernel_desc;
  err = npu_runtime_get_kernel_desc(&runtime, "tx_unit_tcsr_write_read", &kernel_desc);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to get kernel description: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  int kernel_ret;

  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 0);

  std::cout << "Kernel return value: " << kernel_ret << std::endl;
  
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to start kernel: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  std::cout << "Kernel run successfully." << std::endl;

  test_dump_mem(&runtime.dev_state, runtime.kernel_set_desc.perf, 0x10);  // profiler
  test_dump_mem(&runtime.dev_state, 0x81000000, 0x100); // args tester

  npu_runtime_deinit(&runtime);


  if(kernel_ret != 0) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }

  return 0;
}
