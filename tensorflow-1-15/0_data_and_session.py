import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node1)
print(node2)
print(node3)

sess = tf.Session()
sess.run([node1, node2, node3])

print(node1)
print(node2)
print(node3)

w = tf.Variable(.3)

sess.run(w)