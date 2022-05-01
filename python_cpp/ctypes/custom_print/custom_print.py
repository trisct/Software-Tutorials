from ctypes import cdll
# mylib = cdll.LoadLibrary("custom_print.so")
mylib = cdll.LoadLibrary("custom_print.so")
mylib.print_hello()