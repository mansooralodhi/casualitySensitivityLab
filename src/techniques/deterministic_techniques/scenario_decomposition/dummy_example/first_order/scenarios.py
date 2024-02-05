
from src.techniques.deterministic_techniques.scenario_decomposition.dummy_example.first_order.sample_data import \
    SampleData
from src.techniques.deterministic_techniques.scenario_decomposition.dummy_example.first_order.sample_model import \
    SampleModel
import numpy as np
from math import factorial


class Scenarios(SampleData, SampleModel):

    def __init__(self, order=2, negativeFeatureInfluence=False):

        SampleData.__init__(self)
        SampleModel.__init__(self)

        # total combinations = ( n! / (r!(n-r)!) )  +  n

        self.order = order
        self.negativeFeatureInfluence = negativeFeatureInfluence
        self.total_combinations = int(
            factorial(self.len_feature) / (factorial(order) * factorial(self.len_feature - order))) + self.len_feature

        self.mask = np.zeros((self.total_combinations, self.len_feature))
        self.x_scenarios = np.zeros((self.total_combinations, self.len_feature))
        self.y_scenarios = np.zeros((self.total_combinations, 1))

    def generate_mask(self):

        identity = np.eye(self.len_feature, self.len_feature)
        self.mask = np.zeros((self.total_combinations - self.len_feature, self.len_feature))
        self.mask = np.vstack([identity, self.mask])
        row = self.len_feature

        for i in range(self.total_combinations):
            for j in range(i + 1, self.len_feature):
                self.mask[row, i] = 1
                self.mask[row, j] = 1
                row += 1

    def generate_scenarios(self):

        referenceFeatVector = self.get_baseFeatureVector()  # (1*n)
        referenceFeatMatrix = np.tile(referenceFeatVector, (self.total_combinations, 1))  # (total_combinations * n)

        alternativeFeatVector = self.get_minFeatureVector() if self.negativeFeatureInfluence else self.get_maxFeatureVector()  # (1*n)
        alternativeFeatMatrix = np.tile(alternativeFeatVector, (self.total_combinations, 1))  # (total_combinations*n)

        self.x_scenarios = (alternativeFeatMatrix * self.mask) + (referenceFeatMatrix * (1 - self.mask))

    def compute_scenarios(self):

        baseCaseOutput = self.calculate_profit(*self.get_baseFeatureVector())

        for i in range(self.total_combinations):

            """ todo: replace the below 'calculate_profit' with any other engineering model. """
            self.y_scenarios[i] = self.calculate_profit(*self.x_scenarios[i, :]) - baseCaseOutput

            if sum(self.mask[i]) == 2:
                # subtracting the individual effects for combination of order 2
                for ind in np.nonzero(self.mask[i])[0]:
                    # y_scenarios[ind]: since we know that first n rows are identity and individual impact
                    self.y_scenarios[i] = self.y_scenarios[i] - self.y_scenarios[ind]


if __name__ == "__main__":
    scenarioDecomposition = Scenarios(order=1, negativeFeatureInfluence=False)
    scenarioDecomposition.generate_mask()
    scenarioDecomposition.generate_scenarios()
    scenarioDecomposition.compute_scenarios()

    print("************  Mask ******************")
    print(scenarioDecomposition.mask)
    print("************  X - Scenarios *********")
    print(scenarioDecomposition.x_scenarios)
    print("************  Y - Scenarios *********")
    print(scenarioDecomposition.y_scenarios)
