
int main() {
    volatile int* ptr = (int*)0x81000000; // Example address, adjust as needed
    (*ptr) = 0x0;
    for(int i = 0 ; i < (0x1000000 / sizeof(int)); i++) {
        ptr[i] = i;
    }
    ptr = (int*)0x82000000;
    (*ptr) = 0;
    while(1) {
        (*ptr)++;
    }
}
