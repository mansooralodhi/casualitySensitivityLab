from src.techniques.deterministic_techniques.scenario_decomposition.first_order.sensitivity_indices import \
    SensitivityIndices
from src.techniques.visual_analytics.gernalized_tornado_diagram import GeneralizedTornadoDiagram
from src.data import *


class ScenarioDecomposition(SensitivityIndices, GeneralizedTornadoDiagram):
    def __init__(self, dataset_name, order: int,
                 negativeFeatureInfluence: bool, onnx_file_path: str):

        dataset = eval(dataset_name)(scale_data=True)

        SensitivityIndices.__init__(self, dataset.X, len(dataset.feature_names),
                                    onnx_file_path, order, negativeFeatureInfluence)
        GeneralizedTornadoDiagram.__init__(self, dataset.feature_names)

        self.individual_sensitivities = list()
        self.interaction_sensitivities = list()
        self.total_sensitivities = list()

    def compute_sensitivity_indices(self):
        self.generate_mask()
        self.generate_scenarios()
        self.compute_scenarios()

        for i in range(self.n):
            self.individual_sensitivities.append(self.individual_sensitivity_index(i))
            self.interaction_sensitivities.append(self.interaction_sensitivity_index(i))
            self.total_sensitivities.append(self.total_sensitivity_index(i))


if __name__ == "__main__":
    finiteChange = ScenarioDecomposition("CaliforniaHousingDataset",
                                         order=2, negativeFeatureInfluence=False,
                                         onnx_file_path="../../../../onnx/artifacts/non_linear_regression_model.onnx")

    finiteChange.compute_sensitivity_indices()

    finiteChange.plot_sensitivities(finiteChange.individual_sensitivities,
                                    finiteChange.interaction_sensitivities,
                                    finiteChange.total_sensitivities,
                                    "Second-Order Scenario Decomposition",
                                    "artifacts//sensitivity_measure.png")
