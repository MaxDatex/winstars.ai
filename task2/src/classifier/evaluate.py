import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.classifier.model import (
    CLASSES,
    IDX2CLASS,
    INFERENCE_TRANSFORM,
    load_trained_model,
)

ROOT: Path     = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = ROOT / "data" / "raw" / "animals10"


def get_test_loader(data_dir: Path, batch_size: int = 32) -> DataLoader:
    ds = ImageFolder(data_dir / "test", transform=INFERENCE_TRANSFORM)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[list[int], list[int], list[float]]:
    """
    Run model on all batches.
    Returns (true_labels, pred_labels, confidences).
    """
    model.to(device)
    model.eval()

    all_true  = []
    all_preds = []
    all_confs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_true.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    return all_true, all_preds, all_confs


def full_evaluation(
    model: nn.Module,
    data_dir: Path = DATA_DIR,
    device: str = "cpu",
) -> float:
    """
    Overall accuracy + per-class precision, recall, F1.
    """
    loader = get_test_loader(data_dir)
    true_labels, pred_labels, _ = collect_predictions(model, loader, device)

    n: int = len(true_labels)
    overall_acc: float = sum(t == p for t, p in zip(true_labels, pred_labels)) / n
    print(f"\nOverall accuracy: {overall_acc:.4f}")

    # per-class stats
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            class_tp[true] += 1
        else:
            class_fp[pred] += 1
            class_fn[true] += 1

    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for idx, cls in enumerate(CLASSES):
        tp = class_tp[idx]
        fp = class_fp[idx]
        fn = class_fn[idx]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support = tp + fn

        print(f"  {cls:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    return overall_acc


def confusion_matrix(
    model: nn.Module,
    data_dir: Path = DATA_DIR,
    device: str = "cpu",
):
    """
    Class x class confusion matrix.
    Rows = true class, cols = predicted class.
    """
    loader = get_test_loader(data_dir)
    true_labels, pred_labels, _ = collect_predictions(model, loader, device)

    n_classes: int = len(CLASSES)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(true_labels, pred_labels):
        matrix[true][pred] += 1

    col_w = 10
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>12}", end="")
    for cls in CLASSES:
        print(f"  {cls[:8]:>{col_w}}", end="")
    print()
    print("  " + "─" * (col_w * (n_classes + 1) + 4))

    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<12}", end="")
        for j in range(n_classes):
            val = matrix[i][j]
            if i == j:
                # correct — green
                print(f"  \033[92m{val:>{col_w}}\033[0m", end="")
            elif val > 0:
                # error — red
                print(f"  \033[91m{val:>{col_w}}\033[0m", end="")
            else:
                print(f"  {val:>{col_w}}", end="")
        print()


def error_analysis(
    model: nn.Module,
    data_dir: Path = DATA_DIR,
    device: str = "cpu",
    max_errors: int = 20,
):
    """
    Wrong predictions with highest confidence.
    """
    loader = get_test_loader(data_dir)
    loader.dataset.samples  # ImageFolder stores (path, label) pairs

    model.to(device)
    model.eval()

    errors = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images_dev = images.to(device)
            logits = model(images_dev)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            for i in range(len(labels)):
                true = labels[i].item()
                pred = preds[i].item()
                conf = confs[i].item()

                if true != pred:
                    # get file path from ImageFolder
                    global_idx = batch_idx * loader.batch_size + i
                    if global_idx < len(loader.dataset.samples):
                        img_path = loader.dataset.samples[global_idx][0]
                    else:
                        img_path = "unknown"

                    errors.append({
                        "path":       img_path,
                        "true":       IDX2CLASS[true],
                        "predicted":  IDX2CLASS[pred],
                        "confidence": conf,
                    })

    errors.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"\nTop {min(max_errors, len(errors))} high-confidence errors "
          f"(total errors: {len(errors)}):")
    print(f"  {'True':<12} {'Predicted':<12} {'Confidence':>12}  Path")
    print(f"  {'─'*12} {'─'*12} {'─'*12}  {'─'*40}")

    for err in errors[:max_errors]:
        print(
            f"  {err['true']:<12} {err['predicted']:<12} "
            f"{err['confidence']:>12.4f}  ...{err['path'][-40:]}"
        )

    return errors


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading trained model...")
    model = load_trained_model()

    full_evaluation(model, device=device)
    confusion_matrix(model, device=device)
    error_analysis(model, device=device, max_errors=20)