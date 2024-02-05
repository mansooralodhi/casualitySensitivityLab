import onnxruntime


class OnnxRuntime(object):
    def __init__(self, onnx_filename: str):
        self.onnx_filename = onnx_filename
        self.ort_session = onnxruntime.InferenceSession(self.onnx_filename)

    def get_onnx_runtime_prediction(self, input_x):
        ort_inputs = {self.ort_session.get_inputs()[0].name: self.to_numpy(input_x)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return dict(ort_inputs=ort_inputs, ort_outs=ort_outs)

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    filename = "artifacts/linear_regression_model.onnx"
    OnnxRuntime(filename)
