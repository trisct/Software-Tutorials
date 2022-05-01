#include <cstdio>

extern "C" {
    void print_hello();
}

void print_hello() {
    printf("hello\n");
    return;
}