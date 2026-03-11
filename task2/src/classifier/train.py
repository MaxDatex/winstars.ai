import argparse
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


def train(args=None):
    data_dir = Path(args.data_dir) if args and args.data_dir else DATA_DIR
    batch_size = args.batch_size if args else 32
    p1_epochs = args.phase1_epochs if args else 10
    p2_epochs = args.phase2_epochs if args else 15
    p1_lr = args.phase1_lr if args else 1e-3
    p2_lr = args.phase2_lr if args else 1e-4
    output_dir = Path(args.output_dir) if args and args.output_dir else MODEL_DIR
    device = args.device if args and args.device else None

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
        epochs=p1_epochs,
        lr=p1_lr,
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
        epochs=p2_epochs,
        lr=p2_lr,
        device=device,
        phase_name="Phase 2 (fine-tuning)",
    )
    print(f"\nBest val accuracy (phase 2): {phase2_acc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model → {save_path}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 image classifier"
    )

    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to animals10 directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to save the trained model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for train and eval"
    )
    parser.add_argument(
        "--phase1_epochs", type=int, default=10, help="Epochs for head-only phase"
    )
    parser.add_argument(
        "--phase2_epochs", type=int, default=15, help="Epochs for fine-tuning phase"
    )
    parser.add_argument(
        "--phase1_lr", type=float, default=1e-3, help="Learning rate for phase 1"
    )
    parser.add_argument(
        "--phase2_lr", type=float, default=1e-4, help="Learning rate for phase 2"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda / cpu (auto-detected if not set)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
