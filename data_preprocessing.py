# Functions for preprocessing datasets (from chapter 19).

import numpy as np

import nnfs
nnfs.init()


def scale_pixels(image_array):
    half_max = 127.5
    return (image_array.astype(np.float32) - half_max) / half_max


def reshape_samples_to_1d(samples):
    return samples.reshape(samples.shape[0], -1)


def shuffle_data(samples, labels):
    keys = np.array(range(samples.shape[0]))
    np.random.shuffle(keys)
    return samples[keys], labels[keys]


def preprocess_image_data(X, y, X_test, y_test):
    """Pipeline including all the above functions."""

    X = scale_pixels(X)
    print(f"Training pixel values are now ranging from {X.min()} to {X.max()}.")
    X_test = scale_pixels(X_test)
    print(f"Testing pixel values are now ranging from {X_test.min()} to {X_test.max()}.")

    X = reshape_samples_to_1d(X)
    print(f"Training data shape is now {X.shape}.")
    X_test = reshape_samples_to_1d(X_test)
    print(f"Testing data shape is now {X_test.shape}.")

    X, y = shuffle_data(X, y)
    print(f"The training data are now shuffled, e.g. the first ten labels: {y[:10]}")

    return X, y, X_test, y_test


# from data_preprocessing import preprocess_image_data
# X, y, X_test, y_test = preprocess_image_data(X, y, X_test, y_test)
