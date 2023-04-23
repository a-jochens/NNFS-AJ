#!/usr/bin/env python3

# Chapter 14
# L1 and L2 Regularization

import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import vertical_data, spiral_data

nnfs.init()  # Fix random seed for reproducibility,
             # set float32 as default dtype,
             # and customize np.dot().


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        """Initialize weights, biases and regularization strength."""
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        """Calculate output values from inputs, weights and biases"""
        self.inputs = inputs  # Remember input values
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases 
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

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
    """Calculate data and regularization losses, given model output and ground truth values."""

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
        
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)

        return regularization_loss


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
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                                         1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, initialize them at 0
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights, 
                # it does not exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous updates
            # multiplied by retain factor and update with current gradients.
            weight_updates = (self.momentum * layer.weight_momentums
                              - self.current_learning_rate * layer.dweights) 
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = (self.momentum * layer.bias_momentums
                            - self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (without using momentum)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
        # Update weights and biases (with or without momentum)
        layer.weights += weight_updates
        layer.biases += bias_updates 

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    """Adaptive Stochastic Gradient Descent"""

    # Set (default) hyperparameters
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                                         1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If layer does not contain cache arrays, initialize them at 0
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter updates + normalization with square rooted cache
        layer.weights -= self.current_learning_rate * layer.dweights / (
                         np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.current_learning_rate * layer.dbiases / (
                        np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:
    """Root Mean Square Propagation Stochastic Gradient Descent"""

    # Set (default) hyperparameters
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                                         1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If layer does not contain cache arrays, initialize them at 0
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache according to RMSprop formula
        layer.weight_cache = (self.rho * layer.weight_cache
                              + (1 - self.rho) * layer.dweights**2)
        layer.bias_cache = (self.rho * layer.bias_cache
                            + (1 - self.rho) * layer.dbiases**2)

        # Vanilla SGD parameter updates + normalization with square rooted cache
        layer.weights -= self.current_learning_rate * layer.dweights / (
                         np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases -= self.current_learning_rate * layer.dbiases / (
                        np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    """Adaptive Momentum: RMSprop with momentum SGD"""

    # Set (default) hyperparameters
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, 
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                                         1. + self.decay * self.iterations)

    def update_params(self, layer):

        # If layer does not contain cache arrays, 
        # initialize momentum and cache arrays at 0
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with bias correction.
        layer.weight_momentums = (self.beta_1 * layer.weight_momentums
                                  + (1 - self.beta_1) * layer.dweights) 
        layer.bias_momentums = (self.beta_1 * layer.bias_momentums
                                + (1 - self.beta_1) * layer.dbiases)
        
        # Get corrected momentum: 
        # self.iteration is 0 at first pass and we need to start with 1 here.
        correction_1 = 1 - self.beta_1 ** (self.iterations + 1)
        weight_momentums_corrected = layer.weight_momentums / correction_1
        bias_momentums_corrected = layer.bias_momentums / correction_1

        # Update cache according to RMSprop formula
        layer.weight_cache = (self.beta_2 * layer.weight_cache
                              + (1 - self.beta_2) * layer.dweights**2)
        layer.bias_cache = (self.beta_2 * layer.bias_cache
                            + (1 - self.beta_2) * layer.dbiases**2)

        # Get corrected cache.
        correction_2 = 1 - self.beta_2 ** (self.iterations + 1)
        weight_cache_corrected = layer.weight_cache / correction_2
        bias_cache_corrected = layer.bias_cache / correction_2

        # Vanilla SGD parameter updates + normalization with square rooted cache
        layer.weights -= self.current_learning_rate * weight_momentums_corrected / (
                         np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (
                        np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Create dense layer with 2 input features and 512 output values
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
                             bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create second dense layer with 512 input features (matching output of previous layer) 
# and 3 output values (matching 3 classes in the data)
dense2 = Layer_Dense(512, 3)

# Create softmax classifier's combined loss and activation 
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)


# Training data
X, y = spiral_data(samples=1000, classes=3)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
# plt.show()

# Training loop
for epoch in range(10_001):

    # Perform a forward pass of our training data through these layers
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    
    # Calculate regularization penalty and overall loss
    regularization_loss = (loss_activation.loss.regularization_loss(dense1)
                           + loss_activation.loss.regularization_loss(dense2))
    loss = data_loss + regularization_loss

    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:  # if one-hot encoded
        y = np.argmax(y, axis=1)  # revert to flat encoding 
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f"epoch: {epoch}, acc: {accuracy:.3f},",
              f"loss: {loss:.3f} (data_loss: {data_loss:.3f},",
              f"reg_loss: {regularization_loss:.3f}),", 
              f"lr: {optimizer.current_learning_rate}")

    # Backward pass (backpropagation)
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# Test model performance

# Test data
X_test, y_test = spiral_data(samples=1000, classes=3)

# Forward pass of the test data
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

# Calculate accuracy
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:              # if one-hot encoded
    y_test = np.argmax(y_test, axis=1)  # revert to flat encoding 
accuracy = np.mean(predictions == y_test)
print(f"\nTest accuracy: {accuracy:.3f} \nTest loss: {loss:.3f}")
