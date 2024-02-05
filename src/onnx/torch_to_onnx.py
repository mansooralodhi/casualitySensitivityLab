import torch.onnx
from src.model.architectures import *

"""
It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, 
to turn the model to inference mode. This is required since operators like dropout or batchnorm 
behave differently in inference and training mode.
"""


class ExportOnnx(object):
    def __init__(self, args: dict):
        self.torch_model = None
        self.model_name = args.get("model_name")
        self.batch_size = args.get("batch_size")
        self.feature_dimension = args.get("feature_dimension")
        self.torch_model_path = args.get("torch_model_path")
        self.export_filename = args.get("export_filename")

    def generate_rand_input(self):
        return torch.randn(self.batch_size, self.feature_dimension)

    def load_torch_model(self):
        self.torch_model = eval(self.model_name)(self.feature_dimension)
        self.torch_model.load_state_dict(torch.load(self.torch_model_path))
        self.torch_model.eval()

    def export(self):
        self.load_torch_model()
        torch.onnx.export(self.torch_model,
                          self.generate_rand_input(),
                          self.export_filename,
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}}
                          )


if __name__ == '__main__':
    parameters = dict(batch_size=1,
                      feature_dimension=8,
                      model_name="NonlinearRegressionModel",
                      torch_model_path="../model/artifacts/non_linear_regression_model.pth",
                      export_filename="artifacts/non_linear_regression_model.onnx")
    ExportOnnx(parameters).export()
