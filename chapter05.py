#!/usr/bin/env python3

# Chapter 5
# Calculating Loss

# import matplotlib.pyplot as plt
import numpy as np

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()  # Fix random seed for reproducibility,
             # set float32 as default dtype,
             # and customize np.dot().


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        """Initialize weights and biases"""
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Calculate output values from inputs, weights and biases"""
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:  # for all loss functions

    def calculate(self, output, y):
        """Calculate data and regularization losses, given model output and ground truth values"""
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # Number of samples in batch
        samples = len(y_pred)

        # Clip data to rule out log(0)
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[:, y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_confidences = -np.log(correct_confidences)
        return negative_log_confidences


X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()


# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 3 input features (matching output of previous layer) 
# and 3 output values (matching 3 classes in the data)
dense2 = Layer_Dense(3, 3)

# Create softmax activation (to be used with output layer)
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()


# Perform a forward pass of our training data through these layers
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Show confidence scores for first few samples
print(activation2.output[:5])


# Calculate mean sample loss
loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)

# Calculate accuracy
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:  # if one-hot encoded
    y = np.argmax(y, axis=1)  # revert to flat encoding 
accuracy = np.mean(predictions == y)
print('accuracy:', accuracy)
