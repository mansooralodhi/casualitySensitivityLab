import torch.nn as nn


class LinearRegressionModel(nn.Module):
    """
    Single Perceptron for regression.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.layers(x)
