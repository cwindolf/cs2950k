import tensorflow as tf 
from tqdm import tqdm
import numpy as np 


# *************************************************************************** #
# Parameters

# Data locations
TRAIN_TXT, TEST_TXT = 'train.txt', 'test.txt'

# Model params
EMBED_SIZE = 30
HIDDEN_SIZE = 100

# Training behavior
BATCH_SIZE = 20
EPOCHS = 1
LEARN_RATE = 1e-4


# *************************************************************************** #
# Input processors

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


def toks(file):
    for line in file:
        for tok in line.split():
            yield tok


def batched_bigrams(words_txt, batch_size, indexer):
    # yield 20 words at a time
    with open(words_txt, 'r') as x_words, open(words_txt) as y_words:
        next(y_words)
        X, Y, count = [], [], 0
        for x, y in zip(toks(x_words), toks(y_words)):
            X.append(indexer[x])
            Y.append(indexer[y])
            count += 1
            if count >= batch_size:
                yield X, Y
                X, Y = [], []
                count = 0


def bigrams(words_txt, indexer):
    with open(words_txt, 'r') as words:
        words = words.read().split()
    words = [indexer[word] for word in words]
    return words[:-1], words[1:]


# *************************************************************************** #
# TF helpers


def weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


# *************************************************************************** #
# Neural net graph

def embed(X, n, k):
    '''
    Given placeholder input, make one-hot vecs and embed them w matrix.
    Args:
        X: tf.placeholder for input sentence (word indices)
        n: size of vocabulary
        k: embedding size
    Returns:
        embedded batch
    '''
    # embedding matrix. n by k to multiply with `X_vecs`
    E = weight([n, k])
    # do the embedding
    return tf.nn.embedding_lookup(E, X)


def forward_pass(embedding, n, k, h):
    '''
    Given embeddings, run a forward pass with relu and softmax logits.
    Args:
        embedding: batch passed through embedding matrix
        n: vocab size
        k: embedding size
        h: size of hidden layer
    Return:
        Softmax logits
    '''
    # First fully connected layer with ReLU activations:
    relu_layer = weight([k, h])
    relu_bias  = weight([h])
    relus = tf.nn.relu(
                tf.add(
                    tf.matmul(embedding, relu_layer),
                    relu_bias))
    # Second for softmax
    softmax_layer = weight([h, n])
    softmax_bias  = weight([n])
    logits = tf.nn.softmax(
                tf.add(
                    tf.matmul(relus, softmax_layer),
                    softmax_bias))
    return logits


# *************************************************************************** #
# Loss and training graph

def loss(Y, logits, n):
    '''
    Given correct word ids Y and estimated logits, what's cross entropy loss?
    Args:
        Y: placehoder to be filled with word ids
        logits: softmax'd estimates
        n: vocab size
    Returns:
        cross entropy loss between Y's one-hots and logits
    '''
    Y_vecs = tf.one_hot(Y, n)
    cross_entropy = tf.reduce_mean(
                        -tf.reduce_sum(
                            Y_vecs * logits,
                            reduction_indices=[1]))
    return cross_entropy


def trainer(learning_rate, loss):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)


# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Compute embeddings

    indexer, n = index_corpus(TRAIN_TXT)
    train_batches = batched_bigrams(TRAIN_TXT, BATCH_SIZE, indexer)
    test_X, test_Y = bigrams(TEST_TXT, indexer)

    # *********************************************************************** #
    # Input/Output placeholders

    X = tf.placeholder(tf.int64, shape=[None])
    Y = tf.placeholder(tf.int64, shape=[None])


    # *********************************************************************** #
    # Instantiate graph

    embedding = embed(X, n, EMBED_SIZE)
    feedforward = forward_pass(embedding, n, EMBED_SIZE, HIDDEN_SIZE)
    cross_entropy = loss(Y, feedforward, n)
    train_op = trainer(LEARN_RATE, cross_entropy)

    # *********************************************************************** #
    # Train and Test

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Train
        print('Training...')
        for X_, Y_ in tqdm(train_batches):
            sess.run(train_op, feed_dict={ X: X_, Y: Y_ })

        # Test
        xe = sess.run(cross_entropy, feed_dict={ X: test_X, Y: test_Y })

    print('Test Perplexity:', np.exp(xe))




