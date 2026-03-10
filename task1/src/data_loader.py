from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def load_mnist_dataset(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Downloads, normalizes, and splits the MNIST dataset.
    Returns: x_train, x_test as pd.DataFrame, y_train, y_test as pd.Series
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    mnist = fetch_openml("mnist_784", as_frame=True)

    x = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    # Normalizing data
    x /= 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test
