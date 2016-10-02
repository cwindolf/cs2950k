import struct
import numpy as np

with open('train-images-idx3-ubyte', 'rb') as training_images:
    training_images.read(16)
    images = []
    for _ in range(60000):
        bytes = training_images.read(784)
        images.append(
            np.reshape(np.asarray(struct.unpack('784B', bytes)),
                       (28, 28)))

with open('train-labels-idx1-ubyte', 'rb') as training_labels:
    training_labels.read(8)
    labels = []
    for _ in range(60000):
        byte = training_labels.read(1)
        labels += struct.unpack('1B', byte)

tot66 = { label : 0.0 for label in range(10) }
tot1313 = { label : 0.0 for label in range(10) }
counts = { label : 0.0 for label in range(10)}
for image, label in zip(images, labels):
    tot66[label] += image[6, 6]
    tot1313[label] += image[13, 13]
    counts[label] += 1.0

for label in range(10):
    c = counts[label]
    print('%d: %3.6f, %3.6f' % (label, tot66[label] / c, tot1313[label] / c))
