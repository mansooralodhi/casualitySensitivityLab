import torch
from src.onnx.onnx_runtime import OnnxRuntime


class Inference(OnnxRuntime):
    def __init__(self, onnx_filename):
        OnnxRuntime.__init__(self, onnx_filename)

    def get_base_case_prediction(self, base_case):
        x = torch.tensor(base_case, dtype=torch.float32)
        x = x.reshape(1, x.shape[0])
        y = self.get_onnx_runtime_prediction(x)['ort_outs'][0]
        return y

    def get_boundary_predictions(self, factor_matrix):
        x = torch.tensor(factor_matrix, dtype=torch.float32)
        y = self.get_onnx_runtime_prediction(x)['ort_outs'][0]
        return y
