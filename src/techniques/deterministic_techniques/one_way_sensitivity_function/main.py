from src.techniques.visual_analytics.siderplots import SiderPlots
from src.techniques.deterministic_techniques.one_way_sensitivity_function.inference import Inference
from src.techniques.deterministic_techniques.one_way_sensitivity_function.predetermined_range import \
    PredeterminedRange
from src.data import *


class OneWaySensitivityFunction(PredeterminedRange, Inference, SiderPlots):
    def __init__(self, num_var_samples, dataset_name, onnx_filename):
        dataset = eval(dataset_name)(scale_data=True)
        self.var_names = dataset.feature_names

        PredeterminedRange.__init__(self, dataset.X, num_var_samples)
        Inference.__init__(self, onnx_filename)
        SiderPlots.__init__(self, self.var_names)

    def calculate_influence(self):
        predetermined_range = self.get_samples()
        return self.run_predetermined_range(predetermined_range["variables_samples"])


if __name__ == '__main__':
    sensitivity_measures = OneWaySensitivityFunction(20, "CaliforniaHousingDataset",
                                                     "../../../onnx/artifacts/non_linear_regression_model.onnx")
    predetermined_predictions = sensitivity_measures.calculate_influence()
    # sensitivity_measures.plot_predetermined_predictions(predetermined_predictions, "One-Way Sensitivity Measure",
    #                                                     "artifacts//sensitivity_measure.png")
