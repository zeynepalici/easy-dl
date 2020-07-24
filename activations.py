import numpy as np


def relu(X):
    return np.maximum(0, X)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))
