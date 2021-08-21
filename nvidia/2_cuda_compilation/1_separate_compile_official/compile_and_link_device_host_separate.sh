# this invokes the device and host linker separately
nvcc --device-c a.cu b.cu # this compiles to a.o and b.o, without linking
nvcc --device-link a.o b.o -o link.o # this links two .o files
g++ a.o b.o link.o -o sep_compile_test -L/usr/local/cuda/lib64/ -lcudart