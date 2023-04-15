#!/usr/bin/env python3

# Chapter 10
# Optimizers

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import vertical_data, spiral_data

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
        self.inputs = inputs  # Remember input values
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs  # Remember input values
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    def forward(self, inputs):

        # Remember input values
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):

        # Initialize gradient array with any values
        self.dinputs = np.empty_like(dvalues)

        # Calculate gradients sample-wise
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Reshape output array as one-column matrix
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Assign sample gradient to row in gradient array
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)


class Loss:  # for all loss functions

    def calculate(self, output, y):
        """Calculate data and regularization losses, given model output and ground truth values"""
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # Number of samples in batch
        n_samples = len(y_pred)

        # Clip data to rule out log(0)
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_confidences = -np.log(correct_confidences)
        return negative_log_confidences

    def backward(self, dvalues, y_true):
        
        n_samples, n_labels = dvalues.shape

        # If labels are sparse, one-hot encode them
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]

        # Calculate and normalize gradient
        self.dinputs = -y_true / dvalues
        self.dinputs /= n_samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    """Combined softmax activation and cross-entropy loss for faster backward step"""

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):

        n_samples = len(dvalues)

        # If labels are one-hot encoded, discretize them
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy for modification
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(n_samples), y_true] -= 1
        # Normalize gradient
        self.dinputs /= n_samples


class Optimizer_SGD:
    """Stochastic Gradient Descent"""

    # Set (default) hyperparameters
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


# X, y = vertical_data(samples=100, classes=3)
X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()

# Create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 64 input features (matching output of previous layer) 
# and 3 output values (matching 3 classes in the data)
dense2 = Layer_Dense(64, 3)

# Create softmax classifier's combined loss and activation 
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


optimizer = Optimizer_SGD()

# Training loop
for epoch in range(10_001):

    # Perform a forward pass of our training data through these layers
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    
    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:  # if one-hot encoded
        y = np.argmax(y, axis=1)  # revert to flat encoding 
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f"epoch: {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}")

    # Backward pass (backpropagation)
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

