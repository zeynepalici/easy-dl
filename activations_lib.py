import numpy as np
from math import e


def relu(X):
    return np.maximum(0, X)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(Z):
    t = e ** Z
    A = t / np.sum(t)
    return A


def relu_backward(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


def sigmoid_backward(Z):
    s = sigmoid(Z)
    dZ = s * (1 - s)
    return dZ


def softmax_backward(y_hat, y):
    return y_hat - y
