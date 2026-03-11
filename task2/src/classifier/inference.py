import argparse
from pathlib import Path
from PIL import Image
import torch

from .model import load_trained_model, predict, MODEL_DIR

ROOT: Path = Path(__file__).resolve().parent.parent.parent

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def run_inference(image_path: Path, model, top_k: int = 3) -> dict:
    """
    Run classifier inference on a single image file.

    Returns:
        {
            "path": "path/to/image.jpg",
            "top_class": "dog",
            "confidence": 0.923,
            "predictions": [
                {"class": "dog", "confidence": 0.923},
                {"class": "cat", "confidence": 0.041},
                {"class": "wolf", "confidence": 0.018},
            ]
        }
    """
    image = Image.open(image_path).convert("RGB")
    predictions = predict(image, model, top_k=top_k)

    return {
        "path": str(image_path),
        "top_class": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "predictions": predictions,
    }


def print_result(result: dict, verbose: bool = False):
    print(f"\nImage: {result['path']}")
    print(f"Predicted : {result['top_class']}  ({result['confidence']:.4f})")
    if verbose:
        print("Top-k predictions:")
        for pred in result["predictions"]:
            print(f"{pred['class']:<12} {pred['confidence']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image classifier inference for animal classification"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single image file")
    input_group.add_argument("--dir", type=str, help="Path to a directory of images")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to trained classifier checkpoint directory",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full top-k breakdown with bar chart",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda / cpu (auto-detected if not set)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_path = (
        Path(args.model_dir) / "best_model.pth"
        if args.model_dir
        else MODEL_DIR / "best_model.pth"
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path)
    model.to(device)

    # collect images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        folder = Path(args.dir)
        image_paths = [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        print(f"Found {len(image_paths)} images in {folder}")

    # run inference
    for image_path in image_paths:
        result = run_inference(image_path, model, top_k=args.top_k)
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
