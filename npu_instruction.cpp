#include "npu_instruction.hpp"

int main() {
    long dep_a asm("a6");
    __tx_matmul_subfloat_e4m3_e4m3_bf16<0, 1, 2>(5, 87, 99, dep_a);
    __tx_matmul_subfloat_e4m3_e4m3_bf16<2, 3, 4>(5, 87, 99);
    __tx_matmul_subfloat_e4m3_e4m3_bf16<4, 5, 6>(5, 87, 99);
    __tx_matmul_subfloat_e4m3_e4m3_bf16<7, 0, 1>(5, 87, 99, dep_a);
    return 0;
}
