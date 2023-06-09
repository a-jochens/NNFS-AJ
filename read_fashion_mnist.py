# Functions for reading the previously downloaded dataset for chapter 19.

import os
import cv2
import numpy as np


def load_mnist_dataset(dataset, path):

    # The labels are the subfolder names here.
    labels = os.listdir(os.path.join(path, dataset))

    # Lists for samples and corresponding labels.
    X = []
    y = []

    # Iterate over label folders.
    for label in labels:
        # Iterate over images in label folder.
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), 
                               cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    """Create and return MNIST train and test data."""

    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    return X, y, X_test, y_test


# from read_fashion_mnist import create_data_mnist
# X, y, X_test, y_test = create_data_mnist("C:\\Users\\AJ\\fashion_mnist_images")
