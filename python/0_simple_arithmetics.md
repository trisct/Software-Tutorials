# Python Tutotials: Basic Operations and Data Structures

### Basic arithmetics

In python, you can perform simple arithmetics

```
>>> 3+5 # addition
8
>>> 3-5 # subtraction
-2
>>> 3*5 # multiplication
15
>>> 3/5 # division
0.6
>>> 3//5 # integer division
0
>>> 3%5 # modular arithmetic
3 
```

You can also use names to represent these values.

```
>>> a = 3
>>> b = 5
>>> a+b
8
>>> a-b
-2
>>> a*b
15
>>> a/b
0.6
>>> a//b
0
>>> a%b
3
```

### Import a package

python has a lot of extensions called packages that can do almost anything. For example, if you want more mathematical operations such as sin, cos, you can use the math package.
```
>>> import math
>>> math.pi
3.141592653589793
>>> math.sin(math.pi)
1.2246467991473532e-16
>>> math.cos(math.pi)
-1.0
```

### Use a list

A python list is, well, a list of things. You can use subscripts to access each individual object (starting from 0).

```
>>> a = [3,6,5,9,10]
>>> a[0]
3
>>> a[4]
10
```

A negative index means to count backwards from the end.

```
>>> a[-1]
10
>>> a[-2]
9
``` 
