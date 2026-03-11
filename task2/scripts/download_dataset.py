import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = ROOT / "data" / "raw" / "animals10"

SPLIT_RATIOS = (0.8, 0.1, 0.1)
SEED = 42

EXPECTED_CLASSES = [
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


def load_hf_dataset():
    print("Loading Rapidata/Animals-10 from HuggingFace...")
    ds = load_dataset("Rapidata/Animals-10")

    data = ds["train"]

    print(f"Loaded {len(data)} images")
    print(f"Features: {data.features}")
    return data


def get_label_mapping(data):
    features = data.features
    label_names = features["label"].names
    mapping = {i: name.lower() for i, name in enumerate(label_names)}

    print(f"\nLabel mapping ({'label'}):")
    for idx, name in mapping.items():
        print(f"{idx}: {name}")

    found = set(mapping.values())
    expected = set(EXPECTED_CLASSES)
    if found != expected:
        raise RuntimeError(
            f"Class mismatch.\nExpected: {sorted(expected)}\nFound: {sorted(found)}\n"
        )

    return mapping


def group_by_class(data, label_mapping: dict) -> dict[str, list]:
    """
    Group dataset indices by class name.
    Returns {class_name: [list of dataset items]}.
    """
    groups = defaultdict(list)
    for item in data:
        class_name = label_mapping[item["label"]]
        groups[class_name].append(item)

    for cls, items in sorted(groups.items()):
        print(f"  {cls:<12}: {len(items)} images")

    return dict(groups)


def split_and_save(
    groups: dict[str, list],
    output_dir: Path,
    ratios: tuple = SPLIT_RATIOS,
):
    """
    For each class, shuffle → split → save images as jpg files.
    """
    random.seed(SEED)
    split_names = ("train", "val", "test")

    # create directories
    for split in split_names:
        for cls in groups:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    stats = {}

    for cls, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        splits = {
            "train": items[:n_train],
            "val": items[n_train : n_train + n_val],
            "test": items[n_train + n_val :],
        }

        for split_name, split_items in splits.items():
            dest_dir = output_dir / split_name / cls
            for i, item in enumerate(split_items):
                img = item["image"]
                img_path = dest_dir / f"{cls}_{i:05d}.jpg"
                img.convert("RGB").save(img_path, "JPEG", quality=95)

        stats[cls] = {s: len(its) for s, its in splits.items()}
        print(
            f"train: {stats[cls]['train']}"
            f"val: {stats[cls]['val']}"
            f"test: {stats[cls]['test']}"
        )

    return stats


def main():
    data = load_hf_dataset()
    label_col = get_label_mapping(data)

    print("\nGrouping by class...")
    groups = group_by_class(data, label_col)

    print("\nSplitting and saving images...")
    stats = split_and_save(groups, OUTPUT_DIR)

    total = sum(
        count for class_stats in stats.values() for count in class_stats.values()
    )
    print(f"\nTotal images saved: {total}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
