from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
import pandas as pd


def load_mnist_dataset(
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Downloads, normalizes, and splits the MNIST dataset.
    Returns: x_train, x_test as pd.DataFrame, y_train, y_test as pd.Series
    """

    mnist = fetch_openml('mnist_784')

    x = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    # Normalizing data
    x /= 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test
