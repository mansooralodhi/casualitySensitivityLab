import torch.nn as nn


class NonlinearRegressionModel(nn.Module):
    """
    Multilayer Neural Network for Regression.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x)
