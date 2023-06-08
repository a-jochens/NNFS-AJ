# Functions for preprocessing datasets (from chapter 19).

import numpy as np

import nnfs
nnfs.init()

from read_fashion_mnist import create_data_mnist


def scale_pixels(image_array):
    half_max = 127.5
    return (image_array.astype(np.float32) - half_max) / half_max


def reshape_samples_to_1d(samples):
    return samples.reshape(samples.shape[0], -1)


def shuffle_data(samples, labels):
    keys = np.array(range(samples.shape[0]))
    np.random.shuffle(keys)
    return samples[keys], labels[keys]


X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

X = scale_pixels(X)
X_test = scale_pixels(X_test)

X = reshape_samples_to_1d(X)
X_test = reshape_samples_to_1d(X_test)

X, y = shuffle_data(X, y)

print(X.min(), X.max())
print(X.shape)
print(X_test.shape)
print(y[:10])
