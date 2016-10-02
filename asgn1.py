import struct
import numpy as np

def load_mnist(imagefile, labelfile, count):

    with open(imagefile, 'rb') as image_data:
        image_data.read(16)
        images = []
        for _ in range(count):
            bytes = image_data.read(784)
            image = np.asarray(struct.unpack('784B', bytes),
                               dtype=np.float_)
            image /= 255.0
            images.append(image)

    with open(labelfile, 'rb') as label_data:
        label_data.read(8)
        labels = []
        for _ in range(count):
            byte = label_data.read(1)
            labels += struct.unpack('1B', byte)

    return images, labels


def debug(*args):
    if False:
        print(args)


def softmax(H):
    exps = np.exp(H - np.max(H))
    return exps / np.sum(exps)


def feedforward(I, W, B):
    debug('Image: ', I, I.shape)
    debug('W:', W, W.shape)
    debug('B:', B, B.shape)
    A = W * I[:, np.newaxis]
    debug('A:', A, A.shape)
    H = B + np.sum(A, axis=0)
    debug('H:', H, H.shape)
    SM = softmax(H)
    debug('SM:', SM, SM.shape)
    return SM


def predict(I, W, B):
    P = feedforward(I, W, B)
    debug(P)
    return np.argmax(P)


def error(c, P):
    return -np.log(P[correct])


def backprop(c, I, P):
    debug('Backprop')
    debug('I', I.shape)
    debug('Label:', c)
    debug('P:', P, P.shape)
    # dE/dHj = { 1 - pj if j = c
    #          { -pj    o/w
    dEdHj = P
    dEdHj[c] = P[c] - 1.0
    # dE/dWij = (dE/dHj)(dHj/dWij) = (above^)(I[i])
    dEdWij = np.outer(I, dEdHj)
    debug('dE/dWij', dEdWij, dEdWij.shape)
    debug('max elt:', np.max(dEdWij))
    # dE/dBj = (dE/dHj)(dHj/dBj) = (above^)(1)
    # dEdBj = dEdHj
    return dEdWij, dEdHj

# *************************************************************************** #
if __name__ == '__main__':
    # *********************************************************************** #
    # Constants

    import sys

    TRAIN_COUNT = 60000
    TEST_COUNT = 10000
    # Learning params
    TRAIN_ITERS = 10000
    L = 0.5

    # *********************************************************************** #
    # Load Data

    train_images, train_labels = load_mnist('train-images-idx3-ubyte',
                                            'train-labels-idx1-ubyte',
                                            TRAIN_COUNT)
    test_images,  test_labels  = load_mnist('t10k-images-idx3-ubyte',
                                            't10k-labels-idx1-ubyte',
                                            TEST_COUNT)

    # *********************************************************************** #
    # Initialize weights

    W = np.zeros((784, 10))
    B = np.zeros((10,))

    # *********************************************************************** #
    # Train
    selection = np.random.choice(TRAIN_COUNT, TRAIN_ITERS)
    for i in selection:
        I = train_images[i]
        c = train_labels[i]
        P = feedforward(I, W, B)
        dW, dB = backprop(c, I, P)
        W -= L * dW
        B -= L * dB
        # sys.exit(0)

    # *********************************************************************** #
    # Test

    correct = 0.0
    total = 0.0
    for i in range(TEST_COUNT):
        I = test_images[i]
        c = test_labels[i]
        p = predict(I, W, B)
        if c == p:
            correct += 1.0
        total += 1.0
    print('Accuracy: ', correct / total)

