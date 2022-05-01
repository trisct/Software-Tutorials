from ctypes import *
mylib = cdll.LoadLibrary('mylib.so')
print(mylib.__dict__)