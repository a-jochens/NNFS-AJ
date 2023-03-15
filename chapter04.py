#!/usr/bin/env python3

# Chapter 4
# Activation Functions

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


# Perform a forward pass of our training data through these layers
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Show confidence scores for first few samples
print(activation2.output[:5])
