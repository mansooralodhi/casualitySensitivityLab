
from src.techniques.deterministic_techniques.essentials.base_case import BaseCase
from src.techniques.deterministic_techniques.scenario_decomposition.first_order.inference import Inference
from math import factorial
import numpy as np


class Scenarios(BaseCase, Inference):

    def __init__(self, X: np.ndarray, n: int, onnx_file_path: str, order : int = 2,
                 negativeFeatureInfluence : bool = False):
        if order not in [1, 2]:
            raise Exception ("In this file, combination up-to only 2nd order are implemented")

        # total combinations = ( n! / (r!(n-r)!) )  +  n

        BaseCase.__init__(self)
        Inference.__init__(self, onnx_file_path)

        self.n = n # no. of feature
        self.X = X # train data -> (m, n)
        self.order = order
        self.negativeFeatureInfluence = negativeFeatureInfluence
        self.total_combinations = int(
            factorial(self.n) / (factorial(order) * factorial(self.n - order))) + self.n

        self.mask = np.zeros((self.total_combinations, self.n))
        self.x_scenarios = np.zeros((self.total_combinations, self.n))
        self.y_scenarios = np.zeros((self.total_combinations, 1))

    def generate_mask(self):

        if self.order == 1:
            self.mask = np.eye(self.n, self.n)
            return

        identity = np.eye(self.n, self.n)
        self.mask = np.zeros((self.total_combinations - self.n, self.n))
        self.mask = np.vstack([identity, self.mask])
        row = self.n

        for i in range(self.total_combinations):
            for j in range(i + 1, self.n):
                self.mask[row, i] = 1
                self.mask[row, j] = 1
                row += 1

    def generate_scenarios(self):

        referenceFeatVector = self.get_means(self.X)  # (1*n)
        referenceFeatMatrix = np.tile(referenceFeatVector, (self.total_combinations, 1))  # (total_combinations * n)

        alternativeFeatVector = self.get_lower_bounds(self.X) if self.negativeFeatureInfluence else self.get_upper_bounds(self.X)  # (1*n)
        alternativeFeatMatrix = np.tile(alternativeFeatVector, (self.total_combinations, 1))  # (total_combinations*n)

        self.x_scenarios = (alternativeFeatMatrix * self.mask) + (referenceFeatMatrix * (1 - self.mask))

    def compute_scenarios(self):

        baseCaseOutput = self.run_onnx_model(self.get_means(self.X)).get("instance_output").item()

        for i in range(self.total_combinations):
            self.y_scenarios[i] = self.run_onnx_model(self.x_scenarios[i, :]).get("instance_output").item() \
                                  - baseCaseOutput
            if sum(self.mask[i]) == 2:
                # subtracting the individual effects for combination of order 2
                for ind in np.nonzero(self.mask[i])[0]:
                    # y_scenarios[ind]: since we know that first n rows are identity and individual impact
                    self.y_scenarios[i] = self.y_scenarios[i] - self.y_scenarios[ind]


if __name__ == "__main__":

    from src.primary_model.datasets import CaliforniaHousingDataset
    california_dataset = CaliforniaHousingDataset(split_data=False, scale_data=True)

    scenarioDecomposition = Scenarios(california_dataset.X,
                                      len(california_dataset.feature_names),
                                      "../../../../onnx/artifacts/linear_regression_model.onnx",
                                      order=2, negativeFeatureInfluence=False)

    scenarioDecomposition.generate_mask()
    scenarioDecomposition.generate_scenarios()
    scenarioDecomposition.compute_scenarios()

    print("************  Mask ******************")
    print(scenarioDecomposition.mask)
    print("************  X - Scenarios *********")
    print(scenarioDecomposition.x_scenarios)
    print("************  Y - Scenarios *********")
    print(scenarioDecomposition.y_scenarios)
