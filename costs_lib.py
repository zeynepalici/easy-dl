import numpy as np


def compute_cost_binary(m, Y, Y_hat):
    Y_hat[Y_hat == 1] = 0.999

    cost = -1 / m * np.sum(np.multiply(Y, np.log(Y_hat)) + np.multiply((1 - Y), np.log(1 - Y_hat)))
    cost = np.squeeze(cost)

    return cost


def compute_cost_multi(m, Y, Y_hat):
    Y_hat[Y_hat == 1] = 0.999

    cost = -1 / m * np.sum(np.multiply(Y, np.log(Y_hat)))
    cost = np.squeeze(cost)

    return cost

