#include <cstdio>

extern "C" {
    double ret_float_type(int x);
    void print_constr(const char * str);
}

double ret_float_type(int x) {
    return double(x);
}

void print_constr(const char * str) {
    printf(str);
}