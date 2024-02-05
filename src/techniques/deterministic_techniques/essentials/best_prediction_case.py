"""
The goal of this script is to identify the instance from the dataset that
give's best accuracy or minimum error on trained model.
Note: the best instance selection is subjective and can be change w.r.t to problem.
currently its mean square error.

We will use this instance as a base/reference to perform one-way-sensitivity analysis.
"""
import numpy as np
from src.data import *
from src.onnx.onnx_runtime import OnnxRuntime


class BestPredictionCase(OnnxRuntime):
    def __init__(self, onnx_filepath, dataset_name):
        super().__init__(onnx_filename=onnx_filepath)

        self.dataloader = None
        self.dataset_name = dataset_name

    @staticmethod
    def criterion(actual: np.asarray, predicted: np.asarray):
        return np.mean((predicted - actual) ** 2)

    def get_dataloader(self):
        dataset = eval(self.dataset_name)(split_data=False, scale_data=True)
        dataloader = Dataset(dataset.X, dataset.Y, scale_data=False)
        return dataloader

    def get_best_prediction_case(self):
        minimum_loss = np.inf
        best_instance_x = None
        best_instance_y = None
        dataloader = self.get_dataloader()
        for i, data in enumerate(dataloader):
            x, y = data
            x = x.reshape(1, x.shape[0])
            output = self.get_onnx_runtime_prediction(x)['ort_outs']
            output = output[0]
            y = y.numpy()
            loss = self.criterion(y, output)
            if loss < minimum_loss:
                minimum_loss = loss
                best_instance_x = x
                best_instance_y = y
        return dict(best_instance_x=best_instance_x, best_instance_y=best_instance_y,
                    best_instance_mse=minimum_loss)


if __name__ == '__main__':
    extractor = BestPredictionCase(
        onnx_filepath="../../../onnx/artifacts/linear_regression_model.onnx",
        dataset_name="CaliforniaHousingDataset")
    extractor = extractor.get_best_prediction_case()
    print(extractor['best_instance_x'])
    print(extractor['best_instance_y'])
    print(extractor['best_instance_mse'])
