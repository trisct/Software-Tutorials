nvcc --device-c a.cu b.cu # this compiles to a.o and b.o, without linking
nvcc a.o b.o -o sep_compile_test # this links two .o files