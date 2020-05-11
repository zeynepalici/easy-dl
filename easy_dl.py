import pandas as pd


class EasyDL:
    def __init__(self, filename, layers=0, units=None, activations=None):
        data = pd.read_csv(filename).to_numpy()  # pandas data frame döndürüyor. biz numpy kullanmak istiyoruz.

        self.X = data[:, :len(data[0]) - 1]
        self.Y = data[:, len(data[0]) - 1]

        self.layers = layers
        self.units = units
        self.activations = activations
