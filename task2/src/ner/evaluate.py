import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from seqeval.metrics import classification_report, f1_score
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from .model import (
    load_trained_model,
    predict,
    extract_animal_entities,
)
from ..pipeline.normalization import AnimalNormalizer

ROOT: Path = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR: Path = ROOT / "data" / "splits"
MODEL_DIR: Path = ROOT / "models" / "ner"


def load_split(name: str) -> list[dict]:
    with open(SPLITS_DIR / f"{name}.json") as f:
        return json.load(f)


def get_token_level_preds(
    records: list[dict],
    model: DistilBertForTokenClassification,
    tokenizer: DistilBertTokenizerFast,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Run model on all records, return (true_labels, pred_labels)
    in seqeval format — list of lists of string labels, -100 excluded.
    """
    true_labels = []
    pred_labels = []

    for record in records:
        sentence = record["sentence"]
        gold = record["labels"]  # original string BIO labels

        preds = predict(sentence, model, tokenizer)
        pred_seq = [p["label"] for p in preds]

        true_labels.append(gold)
        pred_labels.append(pred_seq)

    return true_labels, pred_labels


def full_evaluation(
    model: DistilBertForTokenClassification,
    tokenizer: DistilBertTokenizerFast,
    split: str = "test",
) -> dict:
    """
    Run seqeval evaluation on a split.
    Prints full classification report and returns metrics dict.
    """
    print(f"\nEvaluating on '{split}' split...")
    records = load_split(split)

    true_labels, pred_labels = get_token_level_preds(records, model, tokenizer)

    print(classification_report(true_labels, pred_labels))

    return {
        "f1": f1_score(true_labels, pred_labels),
    }


def error_analysis(
    model: DistilBertForTokenClassification,
    tokenizer: DistilBertTokenizerFast,
    split: str = "test",
    max_errors: int = 20,
):
    """
    Print sentences where the model made mistakes.
    Shows exactly which token was wrong and what was predicted.
    """
    print(f"\nError analysis on '{split}' split (max {max_errors} shown):")
    print("─" * 70)

    records = load_split(split)
    error_count = 0

    for record in records:
        if error_count >= max_errors:
            break

        sentence = record["sentence"]
        gold = record["labels"]
        preds = predict(sentence, model, tokenizer)
        pred_seq = [p["label"] for p in preds]
        words = [p["word"] for p in preds]

        gold = gold
        pred_seq = pred_seq

        if gold == pred_seq:
            continue  # correct — skip

        error_count += 1
        print(f"Sentence : {sentence}")
        print(f"{'Token':<20} {'Gold':<14} {'Predicted':<14}")
        for word, g, p in zip(words, gold, pred_seq):
            marker = "  " if g == p else "X"
            print(f"{word:<18} {g:<14} {p:<14} {marker}")
        print()

    print(f"Total errors shown: {error_count}")


def entity_confusion_matrix(
    model: DistilBertForTokenClassification,
    tokenizer: DistilBertTokenizerFast,
    split: str = "test",
):
    """
    Shows a matrix: rows = true canonical, cols = predicted canonical.
    """
    normalizer = AnimalNormalizer()
    records = load_split(split)

    # confusion[true][pred] = count
    confusion = defaultdict(lambda: defaultdict(int))

    for record in records:
        sentence = record["sentence"]
        true_canonical = record["canonical"]  # ground truth class

        preds = predict(sentence, model, tokenizer)
        entities = extract_animal_entities(preds)
        pred_canonical = normalizer.normalize(entities[0]) if entities else None

        confusion[str(true_canonical)][str(pred_canonical)] += 1

    # collect all canonical labels that appear
    all_labels = sorted(
        set(
            list(confusion.keys())
            + [p for preds in confusion.values() for p in preds.keys()]
        )
    )

    # print matrix
    col_w = 12
    print(f"\nEntity confusion matrix:")
    print(f"  {'':>12}", end="")
    for label in all_labels:
        print(f"  {label:>{col_w}}", end="")
    print()
    print("  " + "─" * (col_w * (len(all_labels) + 1) + 2))

    for true_label in all_labels:
        print(f"  {true_label:>{col_w}}", end="")
        for pred_label in all_labels:
            count = confusion[true_label][pred_label]
            marker = (
                f"{count:>{col_w}}"
                if count == 0
                else f"\033[91m{count:>{col_w}}\033[0m"
                if true_label != pred_label
                else f"\033[92m{count:>{col_w}}\033[0m"
            )
            print(f"  {marker}", end="")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading trained model...")
    model, tokenizer = load_trained_model(MODEL_DIR)

    full_evaluation(model, tokenizer, split="test")

    error_analysis(model, tokenizer, split="test", max_errors=20)

    entity_confusion_matrix(model, tokenizer, split="test")
