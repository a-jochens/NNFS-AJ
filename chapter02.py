#!/usr/bin/env python3

# Chapter 2
# Coding Our First Neurons

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = bias + sum(i * w for i, w, in zip(inputs, weights))
print(output)
