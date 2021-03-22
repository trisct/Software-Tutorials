import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# defining computation graph
w = tf.Variable(.3, tf.float32)
b = tf.Variable(-.3, tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

out = w * x + b

loss = tf.reduce_sum(tf.square(out - y))

optimizer = tf.train.GradientDescentOptimizer(1e-3)
train = optimizer.minimize(loss)

# generating training samples
x_train = np.random.random_sample((100,)).astype(np.float32)
y_train = np.random.random_sample((1,)) * x_train + np.random.random_sample((1,)) + .1 * np.random.random_sample((100,)).astype(np.float32)

# initializing variables and session
init = tf.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:x_train, y:y_train})
    current_loss = sess.run(loss, {x:x_train, y:y_train})
    print("step %d: training loss %f" % (i, current_loss))

w_res = sess.run(w)
b_res = sess.run(b)

x_test = np.linspace(x_train.min(), x_train.max(), 100)
y_test = x_test * w_res + b_res

plt.scatter(x_train, y_train)
plt.plot(x_test, y_test)
plt.savefig('1_linear_regression.png')
