from src.techniques.deterministic_techniques.scenario_decomposition.dummy_example.first_order.sample_data import SampleData


class SampleModel(object):
    def __init__(self):
        pass

    @staticmethod
    def calculate_profit(x1, x2, x3):
        return (x1 * x2) + x3


if __name__ == '__main__':
    data = SampleData()
    model = SampleModel()
    y_baseCase = model.calculate_profit(*data.get_baseFeatureVector())
    print(y_baseCase)