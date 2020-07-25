import numpy as np


def relu(X):
    return np.maximum(0, X)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def relu_backward(dA, Z_values):
    dZ = np.array(dA, copy=True)
    dZ[Z_values <= 0] = 0
    return dZ


def sigmoid_backward(dA, Z_values):
    s = 1 / (1 + np.exp(-Z_values))
    dZ = dA * s * (1 - s)
    return dZ
