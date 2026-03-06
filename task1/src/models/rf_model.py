from sklearn.ensemble import RandomForestClassifier

from ..interface import MnistClassifierInterface
from typing import Optional


class RandomForestModel(MnistClassifierInterface):
    def __init__(
            self,
            n_estimators: Optional[int] = 300,
            n_jobs: Optional[int] = 1,
            random_state: Optional[int] = 42,
            **kwargs,
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )

    def train(self, x_train, y_train):
        print("Training...")
        self.rf.fit(x_train, y_train)
        print("Training Complete")

    def predict(self, x_test):
        print("Predicting...")
        return self.rf.predict(x_test)
