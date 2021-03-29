import time

a = 3215.35127
b = 3.

start = time.time()
for i in range(100000000):
    c = a / b
end = time.time()
time_elapsed = end - start
print('Time elapsed (div ver) = %.5f' % time_elapsed)


a = 3215.35127
b = 1./3.
start = time.time()
for i in range(100000000):
    c = a * b
end = time.time()
time_elapsed = end - start
print('Time elapsed (mul ver) = %.5f' % time_elapsed)
