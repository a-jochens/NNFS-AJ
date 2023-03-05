#!/usr/bin/env python3

# Chapter 2
# Coding Our First Neurons

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = [sum(i * w for i, w, in zip(inputs, weights[n])) + biases[n] 
                 for n in range(3)]

print(layer_outputs)
