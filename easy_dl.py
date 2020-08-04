import pandas as pd
import numpy as np
from tqdm import tqdm
import activations_lib


class EasyDL:
    def __init__(self, filename, layers=3, neurons=[4, 6, 1], activations=None, learning_rate=0.1, iterations=1000):
        data = pd.read_csv(filename).to_numpy()

        self.X = data[:, :len(data[0]) - 1]
        self.Y = data[:, len(data[0]) - 1]
        self.Y = self.Y.reshape((self.Y.shape[0], 1))
        self.number_of_examples, self.number_of_features = self.X.shape

        self.layers = layers
        self.neurons = neurons
        self.neurons.insert(0, self.number_of_features)
        self.activations = activations
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.predicted_weights = []
        self.predicted_b_values = []

    def learn(self):
        costs = []

        weights, b_values = self._initialize_parameters()
        if not self.activations:
            self.activations = []
            for _ in range(self.layers - 1):
                self.activations.append("relu")
            self.activations.append("sigmoid")

        for _ in tqdm(iterable=range(self.iterations), desc="Learning"):
            Z_values, A_values = self._forward_prop(self.X, weights, b_values)
            dW_values, db_values = self._backward_prop(weights, Z_values, A_values)
            dW_values.reverse()
            db_values.reverse()

            for i in range(self.layers):
                weights[i] = weights[i] - self.learning_rate * dW_values[i]
                b_values[i] = b_values[i] - self.learning_rate * db_values[i]

            self.predicted_weights = weights
            self.predicted_b_values = b_values

            costs.append(self._compute_cost(self.test()))

        return costs

    def test(self):
        _, A_values = self._forward_prop(self.X, self.predicted_weights, self.predicted_b_values)
        return A_values[-1]

    def predict(self, filename):
        test_data = pd.read_csv(filename).to_numpy()
        _, A_values = self._forward_prop(test_data, self.predicted_weights, self.predicted_b_values)
        return np.round(A_values[-1])

    def _initialize_parameters(self):
        weights = []
        b_values = []

        for i in range(1, self.layers + 1):
            current_weight = np.random.randn(self.neurons[i], self.neurons[i - 1]) * np.sqrt(2 / self.neurons[i - 1])
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

    def _backward_prop(self, weights, Z_values, A_values):
        A_values[-1][A_values[-1] == 1] = 0.999

        dA = -(np.divide(self.Y, A_values[-1]) - np.divide(1 - self.Y, 1 - A_values[-1]))
        dW_values = []
        db_values = []

        for i in reversed(range(self.layers)):
            dW, db, dA_back = self._backward_prop_step(weights[i], Z_values[i], A_values[i], dA, self.activations[i])
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

        return Z, A

    def _backward_prop_step(self, W, Z, A, dA, activation):
        dZ = None
        if activation == "relu":
            dZ = dA * activations_lib.relu_backward(Z)
        elif activation == "sigmoid":
            dZ = dA * activations_lib.sigmoid_backward(Z)

        dW = 1 / self.number_of_examples * np.dot(dZ.T, A)
        db = 1 / self.number_of_examples * np.sum(dZ, axis=0, keepdims=True)
        dA_back = np.dot(dZ, W)

        return dW, db.T, dA_back

    def _compute_cost(self, AL):
        AL[AL == 1] = 0.999

        cost = -1 / self.number_of_examples * np.sum(
            np.multiply(self.Y, np.log(AL)) + np.multiply((1 - self.Y), np.log(1 - AL)))
        cost = np.squeeze(cost)

        return cost
