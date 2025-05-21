#include <stdio.h>

extern long test_tcsr_wr();
int main() {
    printf("Testsuit: tcsr_xchg basic test\n");
    return test_tcsr_wr();
}
