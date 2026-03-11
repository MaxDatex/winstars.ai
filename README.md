# Winstars AI DS internship test

Two independent ML tasks covering OOP design patterns, NLP, and Computer Vision.

---

## Task 1 — MNIST Classification + OOP

Three classifiers for the MNIST digit dataset built behind a unified interface.

**Models implemented:**
- Random Forest — scikit-learn baseline
- Feed-Forward Neural Network — 3-layer MLP
- Convolutional Neural Network — 2-block conv net

**Key design:** each model implements `MnistClassifierInterface` and is hidden behind a single `MnistClassifier` entry point that selects the implementation via an `algorithm` parameter (`"rf"`, `"nn"`, `"cnn"`).

---

## Task 2 — Animal Fact-Checker Pipeline

An ML pipeline that checks whether a text claim about an animal matches an image, returning a boolean.

**Models used:**
- DistilBERT fine-tuned for Named Entity Recognition — extracts the animal entity from free-form text
- EfficientNet-B0 fine-tuned on Rapidata/Animals-10 (~23k images, 10 classes) — classifies the animal in the image

**Key design:** the two models are trained independently and bridged by a normalization layer that maps NER outputs (breeds, synonyms, plurals) to classifier class labels. The pipeline handles varied user phrasings — not just "there is a X in the picture".

---

## Setup

```bash
git clone clone https://github.com/MaxDatex/winstars.ai.git
cd winstars.ai

# Task 1
cd task1
pip install -r requirements.txt

# Task 2
cd task2
pip install -r requirements.txt
```

For full instructions, training commands, and usage examples see the task-specific READMEs:

- [`task1/README.md`](task1/README.md)
- [`task2/README.md`](task2/README.md)