from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class CaliforniaHousingDataset(object):
    """
    Source:
    https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html#:~:text=In%20this%20notebook%2C%20we%20will%20quickly%20present%20the,scikit-learn.%20from%20sklearn.datasets%20import%20fetch_california_housing%20california_housing%20%3D%20fetch_california_housing%28as_frame%3DTrue%29
    """
    def __init__(self, scale_data=True, split_data=False):
        data = fetch_california_housing()
        self.X = data.data
        self.Y = data.target
        self.target_names = data.target_names
        self.feature_names = data.feature_names

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        if scale_data:
            self.scale_dataset()
        if split_data:
            self.split_dataset()

    def scale_dataset(self):
        # scalar = MinMaxScaler().fit(self.X)
        # self.X = scalar.transform(self.X)
        self.X = StandardScaler().fit_transform(self.X)

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)


if __name__ == '__main__':

    housing_data = CaliforniaHousingDataset()

    print(f"X Shape:  {housing_data.X.shape}")
    print(f"Y Shape:  {housing_data.Y.shape}")
    print(f"Features Names:  {housing_data.feature_names}")
    print(f"Target Name:  {housing_data.target_names}")