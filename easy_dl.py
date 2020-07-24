import pandas as pd
import numpy as np


class EasyDL:
    def __init__(self, filename, layers=0, neurons=None, activations=None):
        data = pd.read_csv(filename).to_numpy()  # pandas data frame döndürüyor. biz numpy kullanmak istiyoruz.

        self.X = data[:, :len(data[0]) - 1]
        self.Y = data[:, len(data[0]) - 1]
        self.number_of_features = len(self.X[0])

        self.layers = layers
        self.neurons = neurons
        self.neurons.insert(0, self.number_of_features)
        self.activations = activations

    def learn(self):
        weight, b_values = self._initialize_parameters()
        print(weight)
        print("------------------------------")
        print(b_values)

    def _initialize_parameters(self):
        weights = []
        b_values = []

        for i in range(1, self.layers+1):
            current_weight = np.random.rand(self.neurons[i], self.neurons[i - 1])
            current_b = np.random.rand(self.neurons[i], 1)

            weights.append(current_weight)
            b_values.append(current_b)

        return weights, b_values

    def _forward_propagation(self):
        w = []
        b = []
        a = []
        for i in range(self.layers):
            if i == 0:
                Z = np.dot(w[i], self.X) + b[i]
                a.append()  # function yaz, z yi ve suanki activation ı gönder
                # w[0] :
            else:
                Z = np.dot(w[i], a[i - 1]) + b[i]
