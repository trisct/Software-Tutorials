from ctypes import *
mylib = cdll.LoadLibrary("call_with_c_params.so")

### NOTE need to set argument types and the return type

# case 1: argtypes and restype not set:
# the function does not work as desired and does not throw errors
# seems that if the return type is not set,
# a default return type `c_int` object is constructed and returned (not verified)
x = mylib.ret_float_type(c_int(90))
print(type(x), x)

# case 2: argtypes not set but restype set
# seems OK, but I wouldn't rely on it
mylib.ret_float_type.restype = c_double
x = mylib.ret_float_type(c_int(90))
print(type(x), x)

# case 3: argtypes and restype set correctly
mylib.ret_float_type.argtypes = [c_int]
mylib.ret_float_type.restype = c_double
x = mylib.ret_float_type(c_int(90))     # correct call, works as expected
print(type(x), x)
# mylib.ret_float_type(c_float(90))   # incorrect call, python-side error

# case 4: argtypes and restype set incorrectly
mylib.ret_float_type.argtypes = [c_float]
mylib.ret_float_type.restype = c_double
print(mylib.ret_float_type(c_float(90)))   # would return incorrect result
# mylib.restype = c_int

mystr = b"Hi there\n"
mystrptr = c_char_p(mystr)
mylib.print_constr(mystr)
