# Animal Pipeline

An ML pipeline that fact-checks animal claims against images.

**Input** — a text claim + an image  
**Output** — `True` if the animal mentioned in the text matches what's in the image, `False` otherwise

```
"There is a cow in the picture." + 🐄 → True
"I think this is a dog."         + 🐄 → False
```

---

## How It Works

The pipeline chains two independent models:

**Step 1 — NER (Named Entity Recognition)**  
A fine-tuned DistilBERT model extracts the animal entity from the text.
Handles any phrasing — "I think that's a golden retriever", "could this be a spider?", etc.

**Step 2 — Normalization**  
Maps the raw extracted entity to a canonical class label.
Handles breeds, synonyms, and plurals — "puppy" → "dog", "golden retriever" → "dog", "cattle" → "cow".

**Step 3 — Image Classification**  
A fine-tuned EfficientNet-B0 predicts which of the 10 animal classes appears in the image.

**Step 4 — Comparison**  
If the normalized NER output matches the classifier's top prediction → `True`, otherwise `False`.

---

## Project Structure

```
task2/
│
├── data/
│   ├── raw/                        # Animals-10 image dataset (after download)
│   │   └── animals10/
│   │       ├── train/  val/  test/
│   ├── processed/                  # Generated NER dataset + normalization table
│   │   ├── ner_dataset.json
│   │   └── normalization.json
│   └── splits/                     # NER train/val/test splits
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── src/
│   ├── ner/
│   │   ├── model.py                # DistilBERT token classifier definition
│   │   ├── train.py                # NER training loop (parametrized)
│   │   ├── inference.py            # NER inference script
│   │   └── evaluate.py             # seqeval metrics, error analysis
│   │
│   ├── classifier/
│   │   ├── model.py                # EfficientNet-B0 definition
│   │   ├── train.py                # Classifier training loop (parametrized)
│   │   ├── inference.py            # Classifier inference script
│   │   └── evaluate.py             # Accuracy, confusion matrix
│   │
│   └── pipeline/
│       ├── normalization.py        # Entity normalization / lookup table
│       └── pipeline.py             # Main pipeline (wires everything together)
│
├── scripts/
│   ├── generate_ner_dataset.py     # Generates synthetic BIO-tagged NER data
│   └── download_dataset.py         # Downloads Animals-10 from HuggingFace
│
├── models/
│   ├── ner/                        # Saved NER model checkpoint
│   └── classifier/
│       └── best_model.pth          # Saved classifier weights
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/MaxDatex/winstars.ai.git
cd winstars.ai/task2

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make the project importable

```bash
# from the task2/ root directory
export PYTHONPATH=.             # Windows: set PYTHONPATH=.
```

> All `python -m src.*` commands assume you are in the `task2/` root directory with `PYTHONPATH=.` set.

---

## Demo Notebook

The demo notebook walks through the full pipeline end to end — dataset generation, training, and inference with real examples:

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Running Inference

### NER inference

```bash
# single sentence
python -m src.ner.inference --text "There is a cow in the picture."

# with token-level breakdown
python -m src.ner.inference --text "I think that is a golden retriever." --verbose

# batch from file (one sentence per line)
python -m src.ner.inference --file my_sentences.txt
```

Example output:
```
Text     : There is a cow in the picture.
Entities : ['cow']
Canonical: ['cow']
```

---

### Classifier inference

```bash
# single image
python -m src.classifier.inference --image photo.jpg

# top-5 with confidence bar chart
python -m src.classifier.inference --image photo.jpg --top_k 5 --verbose

# entire folder
python -m src.classifier.inference --dir my_photos/
```

Example output:
```
Image     : photo.jpg
Predicted : cow  (0.9823)
```

---

### Full pipeline

```bash
# boolean output only (task deliverable)
python -m src.pipeline.pipeline \
    --text "There is a cow in the picture." \
    --image photo.jpg

# full breakdown
python -m src.pipeline.pipeline \
    --text "There is a cow in the picture." \
    --image photo.jpg \
    --verbose
```

Example output (verbose):
```
─── Pipeline Result ───────────────────────────────
  Text              : There is a cow in the picture.
  Extracted entity  : ['cow']
  Normalized entity : cow
  Image prediction  : cow (0.9823)
  Match             : True
───────────────────────────────────────────────────
```

### From Python

```python
from PIL import Image
from src.pipeline.pipeline import AnimalPipeline

pipeline = AnimalPipeline()
image    = Image.open("photo.jpg")

result = pipeline.run("There is a cow in the picture.", image)
print(result.match)          # True / False
print(result.reason)         # explanation if False
print(result.image_confidence)  # classifier confidence
```

---

## Evaluation

### NER evaluation

```bash
python -m src.ner.evaluate
```

Outputs seqeval classification report, error analysis, and entity-level confusion matrix.

### Classifier evaluation

```bash
python -m src.classifier.evaluate
```

Outputs per-class precision/recall/F1, confusion matrix, and high-confidence errors.

---

## EDA Notebook

Open the exploratory data analysis notebook to understand the dataset before training:

```bash
jupyter notebook notebooks/eda.ipynb
```

Covers:
- Class distribution across splits
- Image size and aspect ratio distributions
- Sample images per class
- Mean color per class
- Potentially confusable class pairs
- NER dataset summary