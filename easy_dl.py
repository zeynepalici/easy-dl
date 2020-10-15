import pandas as pd
import utils
import costs_lib
import numpy as np
from tqdm import tqdm
import activations_lib


class EasyDL:
    def __init__(self, filename, neurons=[2, 3, 1], activations=None, learning_rate=0.1, epoch=1000,
                 mini_batch_size=256):
        data = pd.read_csv(filename).to_numpy()

        self.X = data[:, :len(data[0]) - 1]
        self.Y = data[:, len(data[0]) - 1]
        self.Y = self.Y.reshape((self.Y.shape[0], 1))
        self.number_of_examples, self.number_of_features = self.X.shape
        self.number_of_classes, self.Y = np.unique(self.Y, return_inverse=True)
        self.multi_class = False

        if len(self.number_of_classes) > 2:
            self.multi_class = True
            neurons[-1] = len(self.number_of_classes)
            self.Y = utils.one_hot(self.Y)

        self.layers = len(neurons)
        self.neurons = neurons
        self.neurons.insert(0, self.number_of_features)
        self.activations = activations
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.mini_batch_size = mini_batch_size
        if self.number_of_examples < 2000:
            self.mini_batch_size = self.number_of_examples

        self.predicted_weights = []
        self.predicted_b_values = []

    def learn(self):
        costs = []

        weights, b_values = self._initialize_parameters()
        if not self.activations:
            self.activations = []
            for _ in range(self.layers - 1):
                self.activations.append("relu")
            if self.multi_class:
                self.activations.append("softmax")
            else:
                self.activations.append("sigmoid")

        mini_batch_num = np.ceil(self.number_of_examples / self.mini_batch_size)
        last_batch_size = self.number_of_examples % self.mini_batch_size

        for _ in tqdm(iterable=range(self.epoch), desc="Learning"):
            for j in range(int(mini_batch_num)):
                start, end = j * self.mini_batch_size, (j + 1) * self.mini_batch_size
                if j == mini_batch_num - 1 and mini_batch_num != 1:
                    start = j * self.mini_batch_size
                    end = start + last_batch_size

                Z_values, A_values = self._forward_prop(self.X[start:end], weights, b_values)
                dW_values, db_values = self._backward_prop(weights, Z_values, A_values, start, end)
                dW_values.reverse()
                db_values.reverse()

                for i in range(self.layers):
                    weights[i] = weights[i] - self.learning_rate * dW_values[i]
                    b_values[i] = b_values[i] - self.learning_rate * db_values[i]

                self.predicted_weights = weights
                self.predicted_b_values = b_values

                if self.multi_class:
                    cost = costs_lib.compute_cost_multi(self.number_of_examples, self.Y, self.evaluate()[0]),
                    costs.append(cost)
                else:
                    cost = costs_lib.compute_cost_binary(self.number_of_examples, self.Y, self.evaluate()[0])
                    costs.append(cost)

        return costs

    def evaluate(self):
        _, A_values = self._forward_prop(self.X, self.predicted_weights, self.predicted_b_values)
        train_accuracy = np.sum(np.round(A_values[-1]) == self.Y) / len(self.Y)
        return A_values[-1], train_accuracy

    def predict(self, filename):
        # TODO: other choices for data other than filename
        test_data = pd.read_csv(filename).to_numpy()
        _, A_values = self._forward_prop(test_data, self.predicted_weights, self.predicted_b_values)
        return np.round(A_values[-1])

    def _initialize_parameters(self):
        weights = []
        b_values = []

        for i in range(1, self.layers + 1):
            current_weight = np.random.randn(self.neurons[i], self.neurons[i - 1]) * np.sqrt(2 / self.neurons[i - 1])
            # he initialization / weight initialization for deep networks
            current_b = np.zeros((self.neurons[i], 1))

            weights.append(current_weight)
            b_values.append(current_b)

        return weights, b_values

    def _forward_prop(self, input_, weights, b_values):
        Z_values = []
        A_values = [input_]
        for i in range(self.layers):
            current_Z, current_A = self._forward_prop_step(A_values[i], weights[i], b_values[i], self.activations[i])
            Z_values.append(current_Z)
            A_values.append(current_A)

        return Z_values, A_values

    def _backward_prop(self, weights, Z_values, A_values, start, end):
        A_values[-1][A_values[-1] == 1] = 0.999

        dA = None
        if not self.multi_class:
            dA = -(np.divide(self.Y[start: end], A_values[-1]) - np.divide(1 - self.Y[start: end], 1 - A_values[-1]))

        dW_values = []
        db_values = []

        for i in reversed(range(self.layers)):
            dW, db, dA_back = self._backward_prop_step(weights[i], Z_values[i], A_values[i], dA, self.activations[i],
                                                       A_values[-1], self.Y[start: end])
            dA = dA_back

            dW_values.append(dW)
            db_values.append(db)

        return dW_values, db_values

    def _forward_prop_step(self, A_back, W, b, activation):
        Z = np.dot(A_back, W.T) + b.T
        A = None
        if activation == "relu":
            A = activations_lib.relu(Z)
        elif activation == "sigmoid":
            A = activations_lib.sigmoid(Z)
        elif activation == "softmax":
            A = activations_lib.softmax(Z)

        return Z, A

    def _backward_prop_step(self, W, Z, A, dA, activation, Y_hat, Y):
        dZ = None
        if activation == "relu":
            dZ = dA * activations_lib.relu_backward(Z)
        elif activation == "sigmoid":
            dZ = dA * activations_lib.sigmoid_backward(Z)
        elif activation == "softmax":
            dZ = activations_lib.softmax_backward(Y_hat, Y)

        dW = 1 / self.number_of_examples * np.dot(dZ.T, A)
        db = 1 / self.number_of_examples * np.sum(dZ, axis=0, keepdims=True)
        dA_back = np.dot(dZ, W)

        return dW, db.T, dA_back
