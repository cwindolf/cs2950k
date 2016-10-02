import tensorflow as tf 
import numpy as np
import struct


# learn rate
LAMBDA = 0.5
# number of examples
TRAIN_COUNT = 60000
TEST_COUNT = 10000
# batch size
BATCH_SIZE = 100
# number of batches to go thru
TRAIN_STEPS = 1000


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


# image placeholder
I = tf.placeholder(tf.float32, (None, 784))
# weights & biases
W = tf.Variable(tf.zeros((784, 10)))
b = tf.Variable(tf.zeros((10,)))
# predictor
y = tf.nn.softmax(tf.matmul(I, W) + b)
# correct answers placeholder
c = tf.placeholder(tf.float32, (None, 10))
# error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(c * tf.log(y), 
                               reduction_indices=[1]))
# training protocol
train_step = tf.train.GradientDescentOptimizer(LAMBDA).minimize(cross_entropy)
# init
init_step = tf.initialize_all_variables()

# load data
train_images, train_labels = load_mnist('train-images-idx3-ubyte',
                                        'train-labels-idx1-ubyte',
                                        TRAIN_COUNT)
test_images,  test_labels  = load_mnist('t10k-images-idx3-ubyte',
                                        't10k-labels-idx1-ubyte',
                                        TEST_COUNT)

# testing
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(c, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # train
    init_step.run()
    for _ in range(TRAIN_STEPS):
        random_selection = np.random.choice(TEST_COUNT, BATCH_SIZE)
        batch_Is = train_images[random_selection]
        batch_cs = train_labels[random_selection]
        train_step.run(feed_dict={I: batch_Is, c: batch_cs})

    # test
    print(sess.run(accuracy, feed_dict={I: test_images, c: test_labels}))
