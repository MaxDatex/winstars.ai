from sklearn.ensemble import RandomForestClassifier

from ..interface import MnistClassifierInterface


class RandomForestModel(MnistClassifierInterface):
    def __init__(
            self,
            n_estimators: int = 300,
            n_jobs: int = 1,
            random_state: int = 42,
            **kwargs,
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )

    def train(self, x_train, y_train):
        self.rf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.rf.predict(x_test)
