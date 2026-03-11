from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

ROOT: Path = Path(__file__).resolve().parent.parent.parent
MODEL_DIR: Path = ROOT / "models" / "classifier"

CLASSES: list[str] = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]
CLASS2IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX2CLASS: dict[int, str] = {idx: cls for idx, cls in enumerate(CLASSES)}

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_model(num_classes: int = 10) -> nn.Module:
    """
    Load EfficientNet-B0 with pretrained ImageNet weights.
    Replace the final classifier for our num_classes.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_backbone(model: nn.Module, layers: int = 1) -> nn.Module:
    """
    Unfreeze the last N blocks of EfficientNet-B0 for fine-tuning.
    """
    total_blocks = len(model.features)
    unfreeze_from = total_blocks - layers

    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total: int = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)"
    )

    return model


def load_trained_model(
    checkpoint_path: Path = MODEL_DIR / "best_model.pth",
) -> nn.Module:
    """
    Load a trained classifier from a .pth checkpoint.
    """
    model = get_model()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(
    image: Image.Image,
    model: nn.Module,
    device: str = "cpu",
    top_k: int = 3,
) -> list[dict]:
    model.to(device)
    model.eval()

    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)  # (1, num_classes)
        probs = probs.squeeze(0)  # (num_classes,)

    top_probs, top_idxs = torch.topk(probs, k=top_k)

    return [
        {
            "class": IDX2CLASS[idx.item()],
            "confidence": prob.item(),
        }
        for prob, idx in zip(top_probs, top_idxs)
    ]


if __name__ == "__main__":
    print("Building model...")
    model = get_model()

    total = sum(p.numel() for p in model.parameters())

    print(f"Total params: {total:,}")
