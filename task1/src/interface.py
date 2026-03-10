from abc import ABC, abstractmethod

import pandas as pd


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model using the provided data
        """
        pass

    @abstractmethod
    def predict(self, x_test: pd.DataFrame):
        """
        Predict using the trained model
        """
        pass
