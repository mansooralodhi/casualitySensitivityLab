import numpy as np
from src.techniques.deterministic_techniques.essentials.base_case import BaseCase


# todo: perform left-to-right step increment of variables samples rather than only forward/right step increment.


class PredeterminedRange(BaseCase):
    def __init__(self, X, num_variable_samples):
        BaseCase.__init__(self)
        self.X = X
        self.num_variable_samples = num_variable_samples

    def get_samples(self):
        variables_lower_limit = self.get_lower_bounds(self.X)
        variables_upper_limit = self.get_upper_bounds(self.X)
        variables_step_size = (variables_upper_limit - variables_lower_limit) / self.num_variable_samples # (1, 8)

        base_case = self.get_means(self.X)
        variables_samples = np.tile(base_case, (self.num_variable_samples, 1)) # (20, 8)
        variables_samples = np.repeat(variables_samples[np.newaxis, :, :], self.X.shape[1], axis=0) # (8, 20, 8)

        for i in range(1, self.X.shape[1]):
            variables_samples[i, 0, i] = variables_lower_limit[0, i]
            for j in range(1, self.num_variable_samples):
                variables_samples[i, j, i] = variables_samples[i, j-1, i] + variables_step_size[0, i]

        return dict(variables_samples=variables_samples,
                    variables_lower_limit=variables_lower_limit,
                    variables_upper_limit=variables_upper_limit)


if __name__ == '__main__':
    from src.primary_model.datasets import CaliforniaHousingDataset

    california_dataset = CaliforniaHousingDataset(split_data=False, scale_data=True)

    bounds = PredeterminedRange(california_dataset.X, num_variable_samples=10)

    variables_range_data = bounds.get_samples()
    print(variables_range_data['variables_samples'].shape)