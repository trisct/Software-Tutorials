# ctypes

ctypes is a foreign function library for python that provides C data types and allows calling dynamically linked functions.

### Tutorial 1: Custom C++ library function compiling and calling

In the `custom_print` folders there is code for implementing a custom function that prints "hello" to stdout:
- `custom_print.cpp` contains the code for the custom print function, which is compiled to a shared library file `custom_print.so`
- The function must be declared as `extern "C"` to avoid name mangling by `g++` (otherwise python cannot find the function symbol).
- In `custom_print.py`, the shared object is loaded via `ctypes` and the function is called.

### Tutorial 2: Custom function calling with C data types

