import numpy as np


class DifferentialImpMeasure(object):
    def __init__(self):
        pass

    @staticmethod
    def uniform_differential_imp_measure(partial_derivatives_vector):
        """
        The formula used to compute differential importance measure,here, assumes that change in delta_x or step_size
        while computing finite difference is uniform across all the features/variables.
        """
        tot_derivative = np.sum(partial_derivatives_vector)
        return partial_derivatives_vector / tot_derivative


if __name__ == '__main__':
    from src.techniques.deterministic_techniques.differentiation_methods.finite_differentiation.finite_difference_techniques.forward_difference import \
        ForwardDifference

    dataset = "CaliforniaHousingDataset"
    onnx_file = "../../../../onnx/artifacts/linear_regression_model.onnx"
    finite_difference_handler = ForwardDifference(dataset_name=dataset, onnx_file_path=onnx_file)
    partial_derivative_vector = finite_difference_handler.compute_derivatives()

    importance_measure = DifferentialImpMeasure()
    print(importance_measure.uniform_differential_imp_measure(partial_derivative_vector))
