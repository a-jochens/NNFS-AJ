#!/usr/bin/env python3

# Chapter 18
# Model Object

import numpy as np
import matplotlib.pyplot as plt

import nnfs
# from nnfs.datasets import vertical_data, spiral_data
from nnfs.datasets import sine_data

nnfs.init()  # Fix random seed for reproducibility,
             # set float32 as default dtype,
             # and customize np.dot().


class Layer_Input:
    """Pseudo-layer, to be used in training loop."""

    def forward(self, inputs):
        self.output = inputs


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        """Initialize weights, biases and regularization strength."""
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
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


class Layer_Dropout:

    def __init__(self, rate):
        # The rate parameter is input as the dropout rate,
        # but stored as the retention rate.
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        # Generate and save scaled mask.
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values.
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs  # Remember input values
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


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

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:

    def forward(self, inputs):
        """Save input and calculate/save output of the sigmoid function."""
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        """Calculate derivatives from output of the sigmoid function."""
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:
    """Just for program consistency; does not alter the data."""

    def forward(self, inputs):
        """Just remember the values."""
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs


class Loss:  # for all loss functions
    """Calculate data and regularization losses, given model output and ground truth values."""

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss, self.regularization_loss()

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
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


class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0.
        # Clip both sides to not introduce bias.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss.
        sample_losses = -(y_true * np.log(y_pred_clipped)
                          + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):

        n_samples, n_outputs = dvalues.shape

        # Clip gradients to prevent division by 0.
        # Clip both sides to not introduce bias.
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate and normalize gradient.
        self.dinputs = -(y_true / dvalues_clipped
                         - (1 - y_true) / (1 - dvalues_clipped)) / n_outputs
        self.dinputs = self.dinputs / n_samples


class Loss_MeanSquaredError(Loss):
    """L2 loss, for regression models."""

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        n_samples, n_outputs = dvalues.shape
        # Calculate gradient on values.
        self.dinputs = -2 * (y_true - dvalues) / n_outputs
        # Normalize gradient.
        self.dinputs /= n_samples


class Loss_MeanAbsoluteError(Loss):
    """L1 loss, for occasional use in regression models."""

    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        n_samples, n_outputs = dvalues.shape
        # Calculate gradient on values.
        self.dinputs = np.sign(y_true - dvalues) / n_outputs
        # Normalize gradient.
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


class Accuracy:
    """Common accuracy calculation for all model types."""

    def calculate(self, predictions, y):
        """Calculate accuracy from predictions and ground truth values y."""
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy


class Accuracy_Regression(Accuracy):
    """Accuracy calculation for regression models."""

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        """Calculate precision based on ground truth y."""
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """Are the residuals within the precision limits?"""
        return np.abs(predictions - y) < self.precision


class Model:

    def __init__(self):
        # Create a list of network objects.
        self.layers = []

    def add(self, layer):
        """Add a layer (or similar object) to the model."""
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        """Set the loss, optimizer, and accuracy."""
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        """Set properties of previous and next layers for each layer."""

        self.input_layer = Layer_Input()
        n_layers = len(self.layers)
        self.trainable_layers = []
        
        for i, layer in enumerate(self.layers):

            # The first proper layer has the input layer as the previous one.
            if i == 0:
                layer.prev = self.input_layer
                layer.next = self.layers[i + 1]

            # All layers except for the first and the last one.
            elif i < n_layers - 1:
                layer.prev = self.layers[i - 1]
                layer.next = self.layers[i + 1]

            # The last layer has the loss as the next layer/object.
            else:
                layer.prev = self.layers[i - 1]
                layer.next = self.loss
                # Remember the last layer to later call its predictions method.
                self.output_layer_activation = layer

            # If the layer has a weights attribute, it is trainable.
            if hasattr(layer, 'weights'):
                self.trainable_layers.append(layer)

    def train(self, X, y, *, epochs=1, print_every=1):

        # Initialize accuracy object.
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs + 1):
            
            # Perform forward pass through all layers.
            output = self.forward(X)

            # Calculate data loss, regularization penalty, and overall loss.
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy.
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            

    def forward(self, X):
        """Perform a forward pass of data X through the model layers."""

        # Set the output that the first layer is expecting as prev.
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        # Return output of the last layer.
        return layer.output
    





"""        
if not epoch % 100:
    print(f"epoch: {epoch}, acc: {accuracy:.3f},",
        f"loss: {loss:.3f} (data_loss: {data_loss:.3f},",
        f"reg_loss: {regularization_loss:.3f}),", 
        f"lr: {optimizer.current_learning_rate}")

# Backward pass (backpropagation)
loss_function.backward(activation3.output, y)
activation3.backward(loss_function.dinputs)
dense3.backward(activation3.dinputs)
activation2.backward(dense3.dinputs)
dense2.backward(activation2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Update weights and biases.
optimizer.pre_update_params()
optimizer.update_params(dense1)
optimizer.update_params(dense2)
optimizer.update_params(dense3)
optimizer.post_update_params()
"""


# Training data
X, y = sine_data()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg'); plt.show()

# Tolerance bounds for accuracy calculation
accuracy_precision = np.std(y) / 250


model = Model()

# Add layers to the model.
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

# Set loss and optimizer objects.
model.set(loss=Loss_MeanSquaredError(),
          optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3))

model.finalize()
model.train(X, y, epochs=10_000, print_every=100)





    

"""
# Test model performance.

X_test, y_test = sine_data()

# Forward pass of the test data
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()

loss = loss_function.calculate(activation3.output, y_test)

# Calculate test accuracy according to custom precision.
predictions = activation3.output 
accuracy = np.mean(np.abs(predictions - y_test) < accuracy_precision)
print(f"\nTest accuracy: {accuracy:.3f} \nTest loss: {loss:.3f}")
"""
