import tensorflow as tf 
from tqdm import tqdm
import numpy as np 

# *************************************************************************** #
# Parameters

# Data locations
MOBY_TRAIN, MOBY_TEST = 'mobyUNK_train.txt', 'mobyUNK_test.txt'

# Model params
EMBED_SIZE = 50
CELL_SIZE = 256

# Training behavior
BATCH_SIZE = 20
NUM_STEPS = 20
EPOCHS = 20
LEARN_RATE = 1e-4
LR_DECAY = 0.5
KEEP_PROB = 0.5

# *************************************************************************** #
# Data processing

def tokens(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield from line.split()


def index_corpus(words_txt):
    n = 0
    indexer = {}
    # Make index set
    with open(words_txt, 'r') as words:
        for line in words:
            for word in line.split():
                if word not in indexer:
                    indexer[word] = n
                    n += 1
    return indexer, n


def batch_windows(corpus, indexer):
    x_batch, x_window = [], []
    y_batch, y_window = [], []
    windows, steps = 0, 0
    x_gen, y_gen = tokens(corpus), tokens(corpus)
    next(y_gen)
    for x_token, y_token in zip(x_gen, y_gen):
        x_window.append(indexer[x_token])
        y_window.append(indexer[y_token])
        steps += 1
        # done with this window, add it to batch and restart
        if steps >= NUM_STEPS:
            x_batch.append(x_window)
            y_batch.append(y_window)
            windows += 1
            x_window, y_window = [], []
            steps = 0
        # done with this batch, yield and restart
        if windows >= BATCH_SIZE:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
            windows = 0

# *************************************************************************** #
# TF helpers

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

# *************************************************************************** #
# Inference graph

# hidden layer to make logits
# ^
# reshape from 3D to 2D
# ^
# BasicLSTMCell with dynamic_rnn
# ^
# dropout
# ^
# embedding lookup
# ^
# X: batches like [[window], [window], [window], ...]

def embed(X, vocab_size):
    E = weight([vocab_size, EMBED_SIZE])
    return tf.nn.embedding_lookup(E, X)


def dropout(embeddings, keep_prob):
    return tf.nn.dropout(embeddings, keep_prob)


def cell():
    
    return cell, initial_state


def lstm(embeddings):
    cell = tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE)
    init_state = cell.zero_state(BATCH_SIZE, tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell, embeddings,
                                       initial_state=init_state)
    return outputs, state, init_state


def out_layer(lstm_outputs, vocab_size):
    W = weight((CELL_SIZE, vocab_size))
    B = weight((vocab_size,))
    return tf.add(
            B,
            tf.matmul(tf.reshape(
                        lstm_outputs, 
                        (BATCH_SIZE * NUM_STEPS, CELL_SIZE)),
                      W))


# *************************************************************************** #
# Loss and training graph

# adam optimizer
# ^
# sum and divide by batch size
# ^
# sequence loss vs Y
# ^
# logits

def loss(Y, logits):
    log_perps = tf.nn.seq2seq.sequence_loss_by_example(
                                            [logits], 
                                            [tf.reshape(Y, [-1])],
                                            [tf.ones([BATCH_SIZE * NUM_STEPS])])
    return tf.reduce_sum(log_perps) / BATCH_SIZE


def trainer(loss):
    return tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Process data

    indexer, vocab_size = index_corpus(MOBY_TRAIN)

    # *********************************************************************** #
    # Input/Output placeholders

    X = tf.placeholder(tf.int32, shape=[None, NUM_STEPS])
    Y = tf.placeholder(tf.int32, shape=[None, NUM_STEPS])
    keep_prob = tf.placeholder(tf.float32)

    # *********************************************************************** #
    # Build model and training ops

    embeddings = embed(X, vocab_size)
    dropped_embeddings = dropout(embeddings, keep_prob)
    outputs, state, init_state = lstm(dropped_embeddings)
    logits = out_layer(outputs, vocab_size)
    perplexity = loss(Y, logits)
    train = trainer(perplexity)

    # *********************************************************************** #
    # Train
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for e in range(EPOCHS):
            # Training loop
            print('Epoch %d' % e)
            print('Training...')
            this_state = sess.run(init_state)
            for x_batch, y_batch in tqdm(batch_windows(MOBY_TRAIN, indexer)):
                this_state, _ = sess.run([state, train],
                                         feed_dict={
                                             X: x_batch,
                                             Y: y_batch,
                                             keep_prob: KEEP_PROB,
                                             init_state: this_state
                                         })
            # Testing loop
            print('Testing...')
            test_perplexity = 0.0
            this_state = sess.run(init_state)
            for x_batch, y_batch in tqdm(batch_windows(MOBY_TEST, indexer)):
                batch_p, this_state = sess.run([perplexity, state],
                                               feed_dict={
                                                   X: x_batch,
                                                   Y: y_batch,
                                                   keep_prob: 1.0,
                                                   init_state: this_state
                                               })
                test_perplexity += batch_p
            print('Test Perplexity', test_perplexity / NUM_STEPS)
            LEARN_RATE *= LR_DECAY


