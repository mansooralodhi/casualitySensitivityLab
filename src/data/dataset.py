import torch
from sklearn.preprocessing import StandardScaler
from src.data.california_housing import CaliforniaHousingDataset


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y, scale_data=True):
        if scale_data:
            X = StandardScaler().fit_transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


if __name__ == '__main__':

    housing_data = CaliforniaHousingDataset()
    dataset = Dataset(housing_data.X, housing_data.Y)

    print(f"Dataset-X Shape: {dataset.X.shape}")
    print(f"Dataset-X Type: {dataset.X.dtype}")
    print(f"Dataset-y Shape: {dataset.y.shape}")
    print(f"Dataset-X Type: {dataset.y.dtype}")
    print(f"Sample Target: {dataset.y[0]}")
    print(f"Sample Instance: {dataset.X[0]}")
