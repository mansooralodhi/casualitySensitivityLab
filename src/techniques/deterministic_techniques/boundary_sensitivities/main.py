from use_cases import UseCases
from src.data import *
from src.techniques.visual_analytics.tornado_diagram import TornadoDiagram
from src.techniques.deterministic_techniques.boundary_sensitivities.inference import Inference


class BoundarySensitivities(UseCases, Inference, TornadoDiagram):
    def __init__(self, dataset_name, onnx_filename):

        dataset = eval(dataset_name)(scale_data=True)
        self.var_names = dataset.feature_names

        UseCases.__init__(self, dataset.X)
        Inference.__init__(self, onnx_filename)
        TornadoDiagram.__init__(self, self.var_names, preview=True)

        self.lower_boundary_difference = list()
        self.upper_boundary_difference = list()

        self.generate_base_case()
        self.generate_use_cases()

        self.base_case_predictions = self.get_base_case_prediction(self.base_case)
        self.lower_boundary_predictions = self.get_boundary_predictions(self.lower_factor_matrix)
        self.upper_boundary_predictions = self.get_boundary_predictions(self.upper_factor_matrix)

    def calculate_influence(self):
        self.lower_boundary_difference = self.lower_boundary_predictions - self.base_case_predictions
        self.upper_boundary_difference = self.upper_boundary_predictions - self.base_case_predictions

        self.lower_boundary_difference = [i[0] for i in self.lower_boundary_predictions]
        self.upper_boundary_difference = [i[0] for i in self.upper_boundary_predictions]



if __name__ == '__main__':
    sensitivity_measure = BoundarySensitivities(dataset_name="CaliforniaHousingDataset",
                                                onnx_filename="../../../onnx/artifacts/non_linear_regression_model.onnx")
    sensitivity_measure.calculate_influence()
    sensitivity_measure.plot_sensitivity(sensitivity_measure.upper_boundary_difference,
                                         "Upper Bound Sensitivity Measure",
                                         "artifacts/upper_bound_sensitivity.png")