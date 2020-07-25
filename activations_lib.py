import numpy as np


def relu(X):
    return np.maximum(0, X)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def relu_backward(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


def sigmoid_backward(Z):
    s = sigmoid(Z)
    dZ = s * (1 - s)
    return dZ
