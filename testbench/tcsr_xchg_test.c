#include <stdio.h>

extern void __tx_init();
extern long test_tcsr_wr();

int main() {
    __tx_init();
    printf("testsuit: tcsr_xchg basic test\n");
    return test_tcsr_wr();
}
