import numpy as np


class BaseCase(object):
    def __init__(self):
        pass

    @staticmethod
    def get_lower_bounds(X):
        x = X.min(axis=0)
        x = x.reshape(1, x.shape[0])
        return x

    @staticmethod
    def get_lower_quantile(X):
        x = np.percentile(X, q=0.25, axis=0)
        x = x.reshape(1, x.shape[0])
        return x

    @staticmethod
    def get_upper_bounds(X):
        x = X.max(axis=0)
        x = x.reshape(1, x.shape[0])
        return x

    @staticmethod
    def get_upper_quantile(X):
        x = np.percentile(X, q=.75, axis=0)
        x = x.reshape(1, x.shape[0])
        return x

    @staticmethod
    def get_means(X):
        return np.mean(X, axis=0)

    @staticmethod
    def get_random_instance(X):
        n = np.random.randint(0, X.shape[0])
        return X[n]


if __name__=='__main__':
    from src.primary_model.datasets import CaliforniaHousingDataset

    dataset = CaliforniaHousingDataset(split_data=False, scale_data=True)
    base_point_handler = BaseCase()

    print(f"Features Lower Bound : {base_point_handler.get_lower_bounds(dataset.X)}")
    print(f"Features Upper Bound : {base_point_handler.get_upper_bounds(dataset.X)}")
    print(f"Features Mean : {base_point_handler.get_means(dataset.X)}")
    print(f"Random Instance  : {base_point_handler.get_random_instance(dataset.X)}")
