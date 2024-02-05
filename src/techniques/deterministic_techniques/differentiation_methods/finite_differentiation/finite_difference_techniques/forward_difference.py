import copy
from abc import ABC
import numpy as np

from src.techniques.deterministic_techniques.differentiation_methods.finite_differentiation.finite_difference_techniques.abstract_finite_difference import \
    AbstractFiniteDifferences
from src.techniques.deterministic_techniques.differentiation_methods.finite_differentiation.inference import \
    Inference
from src.techniques.deterministic_techniques.differentiation_methods.finite_differentiation.sensitivity_measure import \
    DifferentialImpMeasure
from src.techniques.visual_analytics.tornado_diagram import TornadoDiagram

from src.primary_model.datasets import *


class ForwardDifference(Inference, AbstractFiniteDifferences, ABC):

    def __init__(self, dataset_name, onnx_file_path):

        AbstractFiniteDifferences.__init__(self)
        Inference.__init__(self, onnx_file_path=onnx_file_path)

        self.dataset = eval(dataset_name)(split_data=False, scale_data=False)
        self.no_features = self.dataset.X.shape[1]
        self.partial_derivatives = np.zeros(self.no_features)

    def calculate_difference_quotient(self, index, x, delta_x, g_0):
        x[index] += delta_x
        g = self.run_onnx_model(x)['instance_output'].item()
        x[index] -= delta_x
        quotient = (g - g_0) / delta_x
        return quotient

    def check_convergence(self, delta_g_curr, delta_g_prev):
        if abs(delta_g_curr - delta_g_prev) < self._epsilon:
            return True
        derivative_difference = (abs(delta_g_curr - delta_g_prev) / delta_g_prev)
        return derivative_difference < self._epsilon

    def compute_partial_derivative(self, index, x, delta_x, g_0):
        iteration = 0
        delta_x_s = delta_x
        differ_quotient = self.calculate_difference_quotient(index, x, delta_x_s, g_0)
        while iteration < self._max_iterations:
            delta_x_s /= self._uniform_perturbation_reduction_factor
            if delta_x_s < self._epsilon:
                return differ_quotient
            differ_quotient_s = self.calculate_difference_quotient(index, x, delta_x_s, g_0)
            if self.check_convergence(differ_quotient_s, differ_quotient):
                print(f"Iteration No. {iteration} ")
                return differ_quotient_s
            differ_quotient = differ_quotient_s
            iteration += 1

    def compute_derivatives(self):
        x_0 = self.get_base_point()
        g_0 = self.run_onnx_model(x_0)['instance_output'].item()
        for i in range(self.no_features):
            print(f"Feature No. {i}")
            x_i = copy.deepcopy(x_0)

            # todo: discuss with professor:
            # case I : return sqrt(machine_epsilon)
            # delta_x_i = self.get_perturbation_size(is_proportional_step=False, x_j=x_0[i])
            # case II : return sqrt(machine_epsilon) * x_j
            # delta_x_i = self.get_perturbation_size(is_proportional_step=True, is_scaled=False, x_j=x_0[i])
            # case III : return sqrt(machine_epsilon) * (1 + x_j)
            delta_x_i = self.get_perturbation_size(is_proportional_step=True, is_scaled=False, x_j=x_0[i])

            self.partial_derivatives[i] = self.compute_partial_derivative(i, x_i, delta_x_i, g_0)
        return self.partial_derivatives


if __name__ == '__main__':
    dataset = "CaliforniaHousingDataset"
    onnx_file = "../../../../../onnx/artifacts/linear_regression_model.onnx"
    finite_difference_handler = ForwardDifference(dataset_name=dataset, onnx_file_path=onnx_file)
    partial_derivatives = finite_difference_handler.compute_derivatives()
    print(partial_derivatives)

    sensitivity_measure = DifferentialImpMeasure()
    differential_measure = sensitivity_measure.uniform_differential_imp_measure(partial_derivatives)

    tornado = TornadoDiagram(finite_difference_handler.dataset.feature_names)
    tornado.plot_sensitivity(partial_derivatives.tolist())
    # tornado.plot_sensitivity(differential_measure.tolist())
