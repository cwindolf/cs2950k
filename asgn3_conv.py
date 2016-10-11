import tensorflow as tf 
import numpy as np
import struct


# learn rate
LAMBDA = 1e-4
# number of examples
TRAIN_COUNT = 60000
TEST_COUNT = 10000
# batch size
BATCH_SIZE = 50
# number of batches to go thru
TRAIN_STEPS = 2000


def load_mnist(imagefile, labelfile, count):
    with open(imagefile, 'rb') as image_data:
        image_data.read(16)
        images = np.empty((count, 784), dtype=np.float32)
        for i in range(count):
            bytes = image_data.read(784)
            image = np.asarray(struct.unpack('784B', bytes),
                               dtype=np.float32)
            image /= 255.0
            images[i] = image
    with open(labelfile, 'rb') as label_data:
        label_data.read(8)
        labels = np.empty((count, 10), dtype=np.float32)
        for i in range(count):
            byte = label_data.read(1)
            labels[i] = np.zeros(10)
            labels[i][struct.unpack('1B', byte)[0]] = 1.0
    return images, labels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# image placeholder
I = tf.placeholder(tf.float32, (None, 784))

# First conv layer
# weights
W_conv1 = weight_variable((5, 5, 1, 32))
b_conv1 = bias_variable((32,))
# input reshape
x = tf.reshape(I, (-1, 28, 28, 1))
# do convolution and pooling
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second conv layer
# weights
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# apply conv and pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# First FC layer
# weights
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# reshape and apply
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
# switch to turn dropout on and off
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# FC2 and Softmax Readout
# weights
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# predict!
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# correct answers placeholder
c = tf.placeholder(tf.float32, (None, 10))

# testing
# error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(c * tf.log(y), 
                               reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(c, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# *************************************************************************** #
# training protocol
train_step = tf.train.AdamOptimizer(LAMBDA).minimize(cross_entropy)
# init
init_step = tf.initialize_all_variables()

# load data
train_images, train_labels = load_mnist('train-images-idx3-ubyte',
                                        'train-labels-idx1-ubyte',
                                        TRAIN_COUNT)
test_images,  test_labels  = load_mnist('t10k-images-idx3-ubyte',
                                        't10k-labels-idx1-ubyte',
                                        TEST_COUNT)

with tf.Session() as sess:
    with tf.device('/cpu:0'): # having extreme GPU problems
        # train
        init_step.run()
        for i in range(TRAIN_STEPS):
            random_selection = np.random.choice(TEST_COUNT, BATCH_SIZE)
            batch_Is = train_images[random_selection]
            batch_cs = train_labels[random_selection]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    I: batch_Is, c: batch_cs, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={I: batch_Is, c: batch_cs, keep_prob: 0.5})

        # test
        print(sess.run(accuracy, feed_dict={
            I: test_images, c: test_labels, keep_prob: 1.0}))
