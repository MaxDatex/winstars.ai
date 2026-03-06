class MnistClassifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.model = self._get_model(algorithm)

    def _get_model(self, algorithm):
        if algorithm == 'rf':
            return RandomForestModel()
        if algorithm == 'nn':
            return FeedForwardNN()
        if algorithm == 'cnn':
            return CNN()