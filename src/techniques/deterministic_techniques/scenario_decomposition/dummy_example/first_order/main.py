from src.techniques.deterministic_techniques.scenario_decomposition.dummy_example.first_order.sensitivity_indices import SensitivityIndices
from src.techniques.visual_analytics.gernalized_tornado_diagram import GeneralizedTornadoDiagram


class ScenarioDecomposition(SensitivityIndices, GeneralizedTornadoDiagram):
    def __init__(self, order=2, negativeFeatureInfluence=False):
        SensitivityIndices.__init__(self, order, negativeFeatureInfluence)
        GeneralizedTornadoDiagram.__init__(self, self.feature_names)

        self.individual_sensitivities = list()
        self.interaction_sensitivities = list()
        self.total_sensitivities = list()

    def compute_sensitivity_indices(self):
        self.generate_mask()
        self.generate_scenarios()
        self.compute_scenarios()

        for i, feature in enumerate(self.feature_names):
            self.individual_sensitivities.append(self.individual_sensitivity_index(i))
            self.interaction_sensitivities.append(self.interaction_sensitivity_index(i))
            self.total_sensitivities.append(self.total_sensitivity_index(i))

if __name__ == "__main__":
    finiteChange = ScenarioDecomposition()
    finiteChange.compute_sensitivity_indices()
    finiteChange.plot_sensitivities(finiteChange.individual_sensitivities,
                                    finiteChange.interaction_sensitivities,
                                    finiteChange.total_sensitivities)
