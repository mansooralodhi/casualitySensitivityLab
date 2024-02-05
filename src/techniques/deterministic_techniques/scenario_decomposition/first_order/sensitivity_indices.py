from src.techniques.deterministic_techniques.scenario_decomposition.first_order.scenarios import Scenarios
import numpy as np


class SensitivityIndices(Scenarios):
    def __init__(self, X: np.ndarray, n: int, onnx_file_path: str, order: int = 2,
                 negativeFeatureInfluence : bool = False):
        super().__init__(X, n, onnx_file_path, order, negativeFeatureInfluence)

    def individual_sensitivity_index(self, featureIndex: int) -> np.ndarray:
        """
        @input: all features impact from their individual to nth order
        @return: only the ith feature 1st order finite change sensitivity index
        """
        return self.y_scenarios[np.where((np.sum(self.mask, axis=1) == 1) & (self.mask[:, featureIndex] != 0))].item()

    def interaction_sensitivity_index(self, featureIndex: int):
        """
        @input: all features impact from their individual to nth order
        @return: only the ith feature 1st order interaction effect
        """
        return np.sum(self.y_scenarios[np.where((np.sum(self.mask, axis=1) > 1) & (self.mask[:, featureIndex] != 0))])

    def total_sensitivity_index(self, featureIndex: int) -> np.ndarray:
        """
        @input: all features impact from their individual to nth order
        @return: only the ith feature 1st order total sensitivity index effect
        """
        return np.sum(self.y_scenarios[np.where((np.sum(self.mask, axis=1) >= 1) & (self.mask[:, featureIndex] != 0))])


if __name__ == "__main__":
    from src.primary_model.datasets import CaliforniaHousingDataset
    california_dataset = CaliforniaHousingDataset(split_data=False, scale_data=True)

    indices = SensitivityIndices(california_dataset.X,
                                 len(california_dataset.feature_names),
                                 "../../../../onnx/artifacts/linear_regression_model.onnx",
                                 order=2, negativeFeatureInfluence=False)
    indices.generate_mask()
    indices.generate_scenarios()
    indices.compute_scenarios()
    print("*************  Feature 0 ************* ")
    print("Individual Sensitivity Index: ",  indices.individual_sensitivity_index(0))
    print("Interaction Sensitivity Index: ",  indices.interaction_sensitivity_index(0))
    print("Total Sensitivity Index: ",  indices.total_sensitivity_index(0))
    print("*************  Feature 1 ************* ")
    print("Individual Sensitivity Index: ", indices.individual_sensitivity_index(1))
    print("Interaction Sensitivity Index: ", indices.interaction_sensitivity_index(1))
    print("Total Sensitivity Index: ", indices.total_sensitivity_index(1))
    print("*************  Feature 2 ************* ")
    print("Individual Sensitivity Index: ", indices.individual_sensitivity_index(2))
    print("Interaction Sensitivity Index: ", indices.interaction_sensitivity_index(2))
    print("Total Sensitivity Index: ", indices.total_sensitivity_index(2))
