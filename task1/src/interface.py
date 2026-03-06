from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the model using the provided data
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """
        Predict using the trained model
        """
        pass
