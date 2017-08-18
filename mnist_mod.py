import random
import image
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import scipy.ndimage

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict= {x: batch_xs, y_: batch_ys})

print ("done with training")


data = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("abc.png", flatten=True)))
result = sess.run(tf.argmax(y,1), feed_dict={x: [data]})

print (' '.join(map(str, result)))