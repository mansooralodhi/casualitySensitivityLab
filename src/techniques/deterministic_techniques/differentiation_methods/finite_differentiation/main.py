from src.data import *
from src.techniques.visual_analytics.tornado_diagram import TornadoDiagram
from src.techniques.deterministic_techniques.differentiation_methods.finite_differentiation.finite_difference_techniques import *
from src.techniques.deterministic_techniques.differentiation_methods.finite_differentiation.sensitivity_measure import DifferentialImpMeasure


class FiniteDifferences(DifferentialImpMeasure, TornadoDiagram):
    def __init__(self, dataset_name, onnx_filename, finite_technique_name):

        self.finite_method = eval(finite_technique_name)(dataset_name, onnx_filename)
        TornadoDiagram.__init__(self, self.finite_method.dataset.feature_names)

        sensitivities = self.compute_sensitivity()
        sensitivity_measures = self.compute_sensitivity_measure(sensitivities)

        self.plot_sensitivity(sensitivity_measures)

    def compute_sensitivity(self):
        return self.finite_method.compute_derivatives()

    def compute_sensitivity_measure(self, partial_derivatives_vector):
        return self.uniform_differential_imp_measure(partial_derivatives_vector)


if __name__=="__main__":
    dataset = "CaliforniaHousingDataset"
    finite_technique = "ForwardDifference"
    onnx_file = "../../../../onnx/artifacts/linear_regression_model.onnx"

    FiniteDifferences(dataset, onnx_file, finite_technique)