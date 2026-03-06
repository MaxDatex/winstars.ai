# MNIST Multi-Model Classification System

This project implements a robust and extensible system for classifying the MNIST dataset using three different machine learning approaches. The architecture is built on the Strategy Design Pattern, ensuring that different models can be swapped in and out with zero friction for the end-user.

### Directory structure
```bash
task1
├── src
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn_model.py
│   │   ├── nn_model.py
│   │   └── rf_model.py
│   ├── __init__.py
│   ├── classifier.py
│   ├── data_loader.py
│   └── interface.py
├── README.md
└── requirements.txt
```

## 🏗 Architecture & Design

The core of this project is the MnistClassifierInterface. By enforcing this interface, we ensure that regardless of whether the model is a simple Scikit-learn ensemble or a complex PyTorch Convolutional Neural Network, the high-level API remains consistent:

1. **Interface** (interface.py): Defines the contract (train and predict).
2. **Factory/Wrapper** (classifier.py): The MnistClassifier class handles model instantiation based on a string parameter (rf, nn, cnn) and manages global random seeds for reproducibility.
3. **Specific Strategies** (models/): Each model handles its own library-specific logic (e.g., reshaping data for CNNs or moving tensors to GPU).

## 🚀 Getting Started

### 1. Installation

Clone the Repository
```bash
git clone https://github.com/MaxDatex/winstars.ai.git
cd mnist_project
```
Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Dependencies
```bash
## Install dependencies
pip install -r requirements.txt
```

### 2. Usage Example

You can switch between rf, nn, and cnn by simply changing the string input in the constructor.

```python
from src.data_loader import load_mnist_dataset
from src.classifier import MnistClassifier

# 1. Load Data
X_train, X_test, y_train, y_test = load_mnist_dataset()

# 2. Initialize (choose: 'rf', 'nn', or 'cnn')
clf = MnistClassifier(algorithm='cnn')

# 3. Train
clf.train(X_train, y_train)

# 4. Predict
predictions = clf.predict(X_test)
```


## 🧠 Implementation Details
The system supports three distinct classification strategies, each tailored for the MNIST dataset:

- **Random Forest (rf)**: An ensemble learning method implemented using Scikit-learn. It processes the data in its flattened format (784 features per image) and serves as an excellent baseline for performance and training speed.
- **Neural Network (nn)**: A Feed-Forward Multi-Layer Perceptron built with PyTorch. This model also utilizes flattened input data and includes dropout layers to improve generalization by preventing overfitting.
- **Convolutional Neural Network (cnn)**: A deep learning architecture implemented in PyTorch that captures spatial dependencies in the images. While it receives flattened data via the common interface, it internally reshapes the input to a 2D format (1 channel, 28x28 pixels) to perform convolutional operations.

### Core Features

- **Data Normalization**: Handled in data_loader.py. All pixel values are scaled to a [0, 1] range to ensure stable training for the neural models.
- **Hardware Acceleration**: Both the nn and cnn models are designed to automatically detect and utilize CUDA-enabled GPUs if available, falling back to CPU otherwise.
- **Reproducibility**: The MnistClassifier wrapper centrally manages random seeds for the random, numpy, and torch libraries to ensure that experiments are consistent across different runs.

📊 Demo & Examples

For a visual walkthrough, performance metrics, and a comparison of all three models, refer to the demo.ipynb notebook.