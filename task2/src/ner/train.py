import json
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, precision_score, recall_score

from .model import (
    LABEL2ID,
    LABELS,
    get_tokenizer,
    get_model,
)

ROOT: Path = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR: Path = ROOT / "data" / "splits"
MODEL_DIR: Path = ROOT / "models" / "ner"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> Dataset:
    """
    Load a JSON split file and return a HuggingFace Dataset.
    record: { sentence, tokens, labels, canonical }
    Take only tokens and labels for training
    """
    path = SPLITS_DIR / f"{name}.json"
    with open(path) as f:
        records = json.load(f)

    return Dataset.from_dict(
        {
            "tokens": [record["tokens"] for record in records],
            "labels": [record["labels"] for record in records],
        }
    )


def tokenize_and_align(batch, tokenizer):
    """
    Tokenize a batch of pre-split sentences and align BIO labels
    to the subword tokenization.
    Ignore special tokens and subsequent subwords

    Examples:
        word_ids:       [None, 0, 1, 2, 3, 3, None]
        batch_labels:   ['O', 'O', 'O', 'B-ANIMAL']
        batch_tokens    ['What', 'a', 'beautiful', 'chickens!']
        tokens          ['[CLS]', 'what', 'a', 'beautiful', 'chickens', '!', '[SEP]']
        aligned_labels  [-100, 0, 0, 0, 1, -100, -100]

        [None, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, None]
        ['O', 'O', 'O', 'O', 'B-ANIMAL', 'O', 'O', 'O', 'O', 'O', 'O']
        ['My', 'friend', 'has', 'a', 'puppies', 'and', 'this', 'looks', 'just', 'like', 'it.']
        ['[CLS]', 'my', 'friend', 'has', 'a', 'pup', '##pies', 'and', 'this', 'looks', 'just', 'like', 'it', '.', '[SEP]']
        [-100, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, -100, -100]

        [None, 0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, None]
        ['O', 'B-ANIMAL', 'O', 'O', 'O', 'O', 'O']
        ['A', 'arachnid', 'is', 'sitting', 'in', 'the', 'photo.']
        ['[CLS]', 'a', 'ara', '##ch', '##ni', '##d', 'is', 'sitting', 'in', 'the', 'photo', '.', '[SEP]']
        [-100, 0, 1, -100, -100, -100, 0, 0, 0, 0, 0, -100, -100]
    """
    tokenized = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
    )

    aligned_labels = []

    for i, word_labels in enumerate(batch["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id: int | None = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                # special token — ignore
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # first subword of this word — assign real label
                label_ids.append(LABEL2ID[word_labels[word_id]])
            else:
                # continuation subword — ignore
                label_ids.append(-100)

            previous_word_id: int | None = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def make_compute_metrics(label_list):
    """
    Returns a compute_metrics function for the Trainer.
    Uses seqeval for entity-level F1.
    """

    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds

        # logits: (batch, seq_len, num_labels) → argmax → predicted ids
        predictions = np.argmax(logits, axis=-1)

        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, label_ids):
            true_seq = []
            pred_seq_clean = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    # skip special tokens and continuation subwords
                    continue
                true_seq.append(label_list[label_id])
                pred_seq_clean.append(label_list[pred_id])

            true_labels.append(true_seq)
            pred_labels.append(pred_seq_clean)

        return {
            "f1": f1_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
        }

    return compute_metrics


def train(args=None):
    print("Loading tokenizer and model...")
    tokenizer = get_tokenizer()
    model = get_model()

    print("Loading datasets...")
    train_ds = load_split("train")
    val_ds = load_split("val")
    print(f"train: {len(train_ds)} samples")
    print(f"val: {len(val_ds)} samples")

    print("Tokenizing and aligning labels...")
    train_ds = train_ds.map(
        lambda batch: tokenize_and_align(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "labels"],  # replace with aligned version
    )
    val_ds = val_ds.map(
        lambda batch: tokenize_and_align(batch, tokenizer),
        batched=True,
        remove_columns=["tokens", "labels"],
    )

    # DataCollator pads each batch to the longest sequence in that batch
    data_collator = DataCollatorForTokenClassification(tokenizer)

    output_dir = Path(args.output_dir) if args and args.output_dir else MODEL_DIR

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs if args else 5,
        learning_rate=args.lr if args else 3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio if args else 0.1,
        per_device_train_batch_size=args.train_batch if args else 32,
        per_device_eval_batch_size=args.eval_batch if args else 64,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(LABELS),
    )

    print("Training...")
    trainer.train()

    print(f"Saving best model to {MODEL_DIR}...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    return trainer


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for animal NER")

    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--train_batch", type=int, default=32, help="Train batch size per device"
    )
    parser.add_argument(
        "--eval_batch", type=int, default=64, help="Eval batch size per device"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--splits_dir", type=str, default=None, help="Override splits directory path"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override model output directory"
    )

    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    train(_args)