import torch
from src.model.architectures import *

"""
To perform testing, first remove a chunk of data from training dataset because write now all the data is being
used for training and validation. 
"""

if __name__ == "__main__":
    model_path = "../artifacts/linear_regression_model.pth"
    model = LinearRegressionModel(feature_dim=8)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    output = model(torch.rand((4, 8)))
    print(output)