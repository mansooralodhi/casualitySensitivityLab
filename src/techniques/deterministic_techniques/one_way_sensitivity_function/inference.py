import torch
import numpy as np
from src.onnx.onnx_runtime import OnnxRuntime


class Inference(OnnxRuntime):
    def __init__(self, onnx_filename):
        OnnxRuntime.__init__(self, onnx_filename)

    def run_predetermined_range(self, predetermined_range):
        x = torch.tensor(predetermined_range, dtype=torch.float32)
        y = np.zeros((x.shape[0], x.shape[1])) # (8, 20)
        for i in range(x.shape[0]):
            x_i = x[i].reshape(x.shape[1], x.shape[2])
            y[i] = self.get_onnx_runtime_prediction(x_i)['ort_outs'][0].flatten()
        return dict(x=x.numpy(), y=y)


if __name__ == "__main__":
    model = Inference("../../../onnx/artifacts/linear_regression_model.onnx")
