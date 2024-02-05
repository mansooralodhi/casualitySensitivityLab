
class SampleData(object):
    def __init__(self):
        self.feature_names = ["x1", "x2", "x3"]
        self.len_feature = len(self.feature_names)

    @staticmethod
    def get_baseFeatureVector():
        return [0.5, 0.5, 0.5]

    @staticmethod
    def get_minFeatureVector():
        return [0, 0, 0]
        # return [-1, -1, -1]

    @staticmethod
    def get_maxFeatureVector():
        return [1, 1, 1]
        # return [2, 2, 2]

