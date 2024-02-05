import onnx
import numpy as np
from src.onnx.torch_to_onnx import ExportOnnx
from src.onnx.onnx_runtime import OnnxRuntime


class CompareOnnxTorch(ExportOnnx, OnnxRuntime):
    def __init__(self, args):
        ExportOnnx.__init__(self, args=args)
        OnnxRuntime.__init__(self, onnx_filename=args.get("export_filename"))

    def _load_onnx(self):
        self.onnx_model = onnx.load(self.onnx_filename)
        onnx.checker.check_model(self.onnx_model)

    def compare_torch_onnx(self):
        # onnx output
        x = self.generate_rand_input()
        ort_outs = self.get_onnx_runtime_prediction(x)['ort_outs']
        # torch output
        self.load_torch_model()
        torch_out = self.torch_model(x)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(self.to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    parameters = dict(batch_size=1,
                      feature_dimension=8,
                      model_name="NonlinearRegressionModel",
                      torch_model_path="../model/artifacts/non_linear_regression_model.pth",
                      export_filename="artifacts/non_linear_regression_model.onnx")
    CompareOnnxTorch(args=parameters).compare_torch_onnx()
