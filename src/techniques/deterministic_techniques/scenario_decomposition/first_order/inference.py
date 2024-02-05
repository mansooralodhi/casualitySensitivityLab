import torch
from src.onnx.onnx_runtime import OnnxRuntime


class Inference(OnnxRuntime):
    def __init__(self, onnx_file_path):
        OnnxRuntime.__init__(self, onnx_filename=onnx_file_path)

    def run_onnx_model(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.reshape(1, x.shape[0])
        y = self.get_onnx_runtime_prediction(x)['ort_outs'][0]
        return dict(instance_input=x, instance_output=y)


if __name__ == "__main__":
    from src.primary_model.datasets import CaliforniaHousingDataset
    from src.techniques.deterministic_techniques.essentials.base_case import \
        BaseCase

    dataset = CaliforniaHousingDataset(split_data=False, scale_data=True)
    base_point_handler = BaseCase()
    base_point = base_point_handler.get_means(dataset.X)

    onnx_file = "../../../../onnx/artifacts/linear_regression_model.onnx"
    model = Inference(onnx_file)
    print(model.run_onnx_model(base_point))
