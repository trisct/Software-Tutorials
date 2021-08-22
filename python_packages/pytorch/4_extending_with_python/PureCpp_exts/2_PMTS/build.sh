# this is for the cpu version of libtorch
# with a minimum number of linking flags 
# note that the CXX11_ABI flag should be set to 0

g++ PMTS.cpp \
-I/home/trisst/.dev_libraries/libtorch-cpu/libtorch/include/ \
-I/home/trisst/.dev_libraries/libtorch-cpu/libtorch/include/torch/csrc/api/include \
-L/home/trisst/.dev_libraries/libtorch-cpu/libtorch/lib/ \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-o PMTS \
-lc10 \
-ltorch_cpu \
-ltorch