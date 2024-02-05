import numpy as np
from src.techniques.deterministic_techniques.essentials.base_case import BaseCase


class UseCases(BaseCase):
    def __init__(self, X: np.ndarray):
        BaseCase.__init__(self)
        self.X = X
        self.base_case = None
        self.lower_factor_matrix = None
        self.upper_factor_matrix = None

    def generate_base_case(self):
        self.base_case = self.get_means(self.X)
        # self.base_case = self.get_random_instance(self.X)

    def generate_use_cases(self):
        factors_upper_bounds = self.get_upper_bounds(self.X)
        factors_lower_bounds = self.get_lower_bounds(self.X)
        tot_factors = self.X.shape[1]
        self.lower_factor_matrix = np.zeros((tot_factors, tot_factors))
        self.upper_factor_matrix = np.zeros((tot_factors, tot_factors))
        for i in range(tot_factors):
            self.lower_factor_matrix[i] = self.base_case
            self.lower_factor_matrix[i][i] = factors_lower_bounds[0][i]
            self.upper_factor_matrix[i] = self.base_case
            self.upper_factor_matrix[i][i] = factors_upper_bounds[0][i]


if __name__ == "__main__":
    from src.primary_model.datasets import CaliforniaHousingDataset

    california_dataset = CaliforniaHousingDataset(split_data=False, scale_data=True)

    mycases = UseCases(california_dataset.X)
    mycases.generate_use_cases()

    print(mycases.lower_factor_matrix)
    print(mycases.upper_factor_matrix)
