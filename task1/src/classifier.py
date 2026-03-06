import random

import numpy as np
import torch

from .interface import MnistClassifierInterface
from .models.rf_model import RandomForestModel
from .models.nn_model import FeedForwardNN
from .models.cnn_model import CNN


class MnistClassifier:
    def __init__(self, algorithm: str, random_state: int = 42):
        self._set_seed(random_state)
        self.model = self._get_model(algorithm)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _get_model(self, algorithm) -> MnistClassifierInterface:
        if algorithm == 'rf':
            return RandomForestModel()
        if algorithm == 'nn':
            return FeedForwardNN()
        if algorithm == 'cnn':
            return CNN()
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Valid options are: 'rf', 'nn', 'cnn'"
        )

    def train(self, x_train, y_train):
        self.model.train(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
