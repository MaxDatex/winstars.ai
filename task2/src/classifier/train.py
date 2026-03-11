import copy
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .model import (
    CLASSES,
    MODEL_DIR,
    TRAIN_TRANSFORM,
    INFERENCE_TRANSFORM,
    get_model,
    unfreeze_backbone,
)

ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = ROOT / "data" / "raw" / "animals10"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Build train/val/test dataloaders from ImageFolder structure.
    """
    train_ds = ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORM)
    val_ds = ImageFolder(data_dir / "val", transform=INFERENCE_TRANSFORM)
    test_ds = ImageFolder(data_dir / "test", transform=INFERENCE_TRANSFORM)

    assert train_ds.classes == CLASSES, (
        f"Dataset classes {train_ds.classes} don't match expected {CLASSES}.\n"
        f"Check that Animals-10 folder names match exactly."
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"train: {len(train_ds):>5} images")
    print(f"val: {len(val_ds):>5} images")
    print(f"test: {len(test_ds):>5} images")

    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    training: bool,
) -> tuple[float, float]:
    """
    Run one epoch
    Returns (avg_loss, accuracy)
    """
    model.train() if training else model.eval()

    total_loss: float = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    avg_loss: float = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    phase_name: str,
) -> tuple[nn.Module, float]:
    """
    Train for a fixed number of epochs with early stopping on val accuracy.
    Returns (best_model, best_val_accuracy).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # only unfreezed params
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc: float = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience = 5
    no_improve = 0

    print(f"\n{phase_name} ({'lr=' + str(lr)})")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        scheduler.step()

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if no_improve >= patience:
            print(
                f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)"
            )
            break

    model.load_state_dict(best_weights)
    return model, best_val_acc


def train(
    data_dir: Path = DATA_DIR,
    batch_size: int = 32,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size)

    model = get_model().to(device)

    model, phase1_acc = train_phase(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=1e-3,
        device=device,
        phase_name="Phase 1 (head only)",
    )
    print(f"\nBest val accuracy (phase 1): {phase1_acc:.4f}")

    print("\nUnfreezing last backbone block...")
    model = unfreeze_backbone(model, layers=1)

    model, phase2_acc = train_phase(
        model,
        train_loader,
        val_loader,
        epochs=15,
        lr=1e-4,
        device=device,
        phase_name="Phase 2 (fine-tuning)",
    )
    print(f"\nBest val accuracy (phase 2): {phase2_acc:.4f}")

    save_path = MODEL_DIR / "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model → {save_path}")

    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, None, device, training=False
    )
    print(f"Test loss : {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return model


if __name__ == "__main__":
    train()
