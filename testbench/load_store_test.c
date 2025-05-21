#include <stdio.h>

#define BUFFER_SIZE_0 128
#define BUFFER_SIZE_1 256

char buffer[BUFFER_SIZE_0 * BUFFER_SIZE_1 * 2];
extern void __tx_init();
extern long test_tx_load_store(char* buffer, long buffer_size, long buffer_size_2d0, long buffer_size_2d1);
int main() {
    __tx_init();
    printf("Testsuit: load_store basic test\n");
    return test_tx_load_store(buffer, BUFFER_SIZE_0 * BUFFER_SIZE_1, BUFFER_SIZE_0, BUFFER_SIZE_1);
}
