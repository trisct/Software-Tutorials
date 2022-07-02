#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <cstdio>

void check_memory(float* ptr, int N, const char* info="\0") {
    if (info[0] != '\0')
        printf("%s: ", info);
    for (int i = 0; i < N; i++)
        printf("%.2f ", ptr[i]);
    printf("\n");
}

void pause_terminal() {
    std::cout << "Press \'Return\' to end." << std::endl;
    std::cin.clear();
    std::cin.get();    
}

template<typename T>
void modulo_affine_init(T* ptr, int N, int a, int b, int m) {
    for (int i = 0; i < N; i++) {
        ptr[i] = T((a * i + b) % m);
    }
}

#endif