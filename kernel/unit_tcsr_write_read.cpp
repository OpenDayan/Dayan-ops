#include "npu_instruction.hpp"
#include "npu_kernel.h"
#include <stdio.h>
#include <string.h>

const int M = 3;
const int N = 4;
const int K = 5;

long dim0;
long dim1;
long dim2;
__device__ long tx_unit_tcsr_write_read(void){
    __tx_unit_tcsr_write<0, 0>(M);  
    __tx_unit_tcsr_write<1, 0>(N);
    __tx_unit_tcsr_write<2, 0>(K); 

    __tx_unit_tcsr_read<0, 0>(dim0);
    __tx_unit_tcsr_read<1, 0>(dim1);
    __tx_unit_tcsr_read<2, 0>(dim2);
    return 0;
}


