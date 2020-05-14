import pandas as pd
import numpy as np


class EasyDL:
    def __init__(self, filename, layers=0, units=None, activations=None):
        data = pd.read_csv(filename).to_numpy()  # pandas data frame döndürüyor. biz numpy kullanmak istiyoruz.

        self.X = data[:, :len(data[0]) - 1]
        self.Y = data[:, len(data[0]) - 1]
        self.features = len(self.X[0])

        self.layers = layers
        self.units = units
        self.activations = activations

    def learn(self):
        pass

    def _initialize_weights(self):
        weights = []
        for i in range(self.layers):
            if i == 0:
                current_weight = np.random.rand(self.units[i], self.features)
            else:
                current_weight = np.random.rand(self.units[i], self.units[i - 1])

            weights.append(current_weight)

        return weights

    def _forward_propagation(self):
        w = []
        b = []
        a = []
        for i in range(self.layers):
            if i == 0:
                Z = np.dot(w[i], self.X) + b[i]
                a.append() #function yaz, z yi ve suanki activation ı gönder
            else:
                Z = np.dot(w[i], a[i-1]) + b[i]
