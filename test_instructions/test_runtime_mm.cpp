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

extern "C" void npu_mm_dump(const struct NPURuntimeState *runtime);
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;


void test_case_0(const struct NPURuntimeState *runtime) {
  printf("\nTestcase 0\n");
  // Alloc 0 1 2
  // Free0   1 2
  // Free1     2
  // Free2

  // Init
  npu_mm_dump(runtime);

  // Malloc
  printf("Start to Malloc 0 1 2..\n");
  uint64_t dev_ptr[64];
  npu_runtime_malloc(runtime, dev_ptr + 0, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 1, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 2, 1024 * 1024); // 1M
  npu_mm_dump(runtime);

  printf("Start to free 0..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[0]);
  npu_mm_dump(runtime);

  printf("Start to free 1..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[1]);
  npu_mm_dump(runtime);

  printf("Start to free 2..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[2]);
  npu_mm_dump(runtime);
}

void test_case_1(const struct NPURuntimeState *runtime) {
  printf("\nTestcase 1\n");
  // Alloc 0 1 2
  // Free1 0   2
  // Free2 0    
  // Free0

  // Init
  npu_mm_dump(runtime);

  // Malloc
  printf("Start to Malloc 0 1 2..\n");
  uint64_t dev_ptr[64];
  npu_runtime_malloc(runtime, dev_ptr + 0, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 1, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 2, 1024 * 1024); // 1M
  npu_mm_dump(runtime);

  printf("Start to free 1..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[1]);
  npu_mm_dump(runtime);

  printf("Start to free 2..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[2]);
  npu_mm_dump(runtime);

  printf("Start to free 0..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[0]);
  npu_mm_dump(runtime);
}
void test_case_2(const struct NPURuntimeState *runtime) {
  printf("\nTestcase 2\n");
  // Alloc 0 1 2
  // Free2 0 1  
  // Free0   1
  // Free1

  // Init
  npu_mm_dump(runtime);

  // Malloc
  printf("Start to Malloc 0 1 2..\n");
  uint64_t dev_ptr[64];
  npu_runtime_malloc(runtime, dev_ptr + 0, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 1, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 2, 1024 * 1024); // 1M
  npu_mm_dump(runtime);

  printf("Start to free 2..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[2]);
  npu_mm_dump(runtime);

  printf("Start to free 0..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[0]);
  npu_mm_dump(runtime);

  printf("Start to free 1..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[1]);
  npu_mm_dump(runtime);
}
void test_case_3(const struct NPURuntimeState *runtime) {
  printf("\nTestcase 3\n");
  // Alloc 0 1 2
  // Free0   1 2 
  // Free2   1  
  // Free1 

  // Init
  npu_mm_dump(runtime);

  // Malloc
  printf("Start to Malloc 0 1 2..\n");
  uint64_t dev_ptr[64];
  npu_runtime_malloc(runtime, dev_ptr + 0, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 1, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 2, 1024 * 1024); // 1M
  npu_mm_dump(runtime);

  printf("Start to free 0..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[0]);
  npu_mm_dump(runtime);

  printf("Start to free 2..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[2]);
  npu_mm_dump(runtime);

  printf("Start to free 1..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[1]);
  npu_mm_dump(runtime);
}
void test_case_4(const struct NPURuntimeState *runtime) {
  printf("\nTestcase 4\n");
  // Alloc 0 1 2
  // Free2 0 1 
  // Free1 0 
  // Free0

  // Init
  npu_mm_dump(runtime);

  // Malloc
  printf("Start to Malloc 0 1 2..\n");
  uint64_t dev_ptr[64];
  npu_runtime_malloc(runtime, dev_ptr + 0, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 1, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 2, 1024 * 1024); // 1M
  npu_mm_dump(runtime);

  printf("Start to free 2..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[2]);
  npu_mm_dump(runtime);

  printf("Start to free 1..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[1]);
  npu_mm_dump(runtime);

  printf("Start to free 0..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[0]);
  npu_mm_dump(runtime);
}

void test_case_5(const struct NPURuntimeState *runtime) {
  printf("\nTestcase 5\n");
  // Alloc 0 1 2
  // Free1 0   2 
  // Free2 0   
  // Free0 

  // Init
  npu_mm_dump(runtime);

  // Malloc
  printf("Start to Malloc 0 1 2..\n");
  uint64_t dev_ptr[64];
  npu_runtime_malloc(runtime, dev_ptr + 0, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 1, 1024 * 1024); // 1M
  npu_runtime_malloc(runtime, dev_ptr + 2, 1024 * 1024); // 1M
  npu_mm_dump(runtime);

  printf("Start to free 1..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[1]);
  npu_mm_dump(runtime);

  printf("Start to free 2..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[2]);
  npu_mm_dump(runtime);

  printf("Start to free 0..\n");
  // Free
  npu_runtime_free(runtime, dev_ptr[0]);
  npu_mm_dump(runtime);
}



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

  test_case_0(&runtime);
  test_case_1(&runtime);
  test_case_2(&runtime);
  test_case_3(&runtime);
  test_case_4(&runtime);
  test_case_5(&runtime);
  
  npu_runtime_deinit(&runtime);

  return 0;
}
