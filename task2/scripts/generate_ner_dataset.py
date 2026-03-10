import json
import random
from pathlib import Path


ROOT: Path = Path(__file__).resolve().parent.parent
PROCESSED: Path = ROOT / "data" / "processed"
SPLITS : Path = ROOT / "data" / "splits"
PROCESSED.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

ANIMAL_VARIANTS: dict[str, list[str]] = {
    "dog": [
        "dog", "dogs", "puppy", "puppies", "canine", "hound", "german shepherd", "golden retriever",
        "labrador", "poodle", "bulldog", "husky", "beagle", "chihuahua"
    ],
    "cat": [
        "cat", "cats", "kitten", "kittens", "feline", "tabby", "siamese", "persian", "kitty"
    ],
    "horse": [
        "horse", "horses", "mare", "stallion", "foal", "pony", "equine", "mustang", "thoroughbred"
    ],
    "spider": ["spider", "spiders", "arachnid", "tarantula", "black widow"],
    "butterfly": ["butterfly", "butterflies", "moth", "caterpillar"],
    "chicken": ["chicken", "chickens", "hen", "rooster", "chick", "poultry"],
    "sheep": ["sheep", "lamb", "lambs", "ewe", "ram", "flock"],
    "cow": ["cow", "cows", "cattle", "bull", "calf", "bovine", "ox"],
    "squirrel": ["squirrel", "squirrels", "chipmunk"],
    "elephant": ["elephant", "elephants", "tusker", "pachyderm"],
}

POSITIVE_TEMPLATES: list[str] = [
    "There is a {animal} in the picture.",
    "There is a {animal} in the image.",
    "There's a {animal} in this photo.",
    "I can see a {animal} in this picture.",
    "I can see a {animal} here.",
    "I see a {animal} in the image.",
    "I see a {animal} here.",
    "This picture shows a {animal}.",
    "This image contains a {animal}.",
    "The image shows a {animal}.",
    "A {animal} is visible in this picture.",
    "A {animal} appears in this photo.",
    "That is a {animal}.",
    "That's definitely a {animal}.",
    "This is a {animal}.",
    "This is clearly a {animal}.",
    "The animal in the picture is a {animal}.",
    "The animal in the photo is a {animal}.",
    "You can see a {animal} in this photo.",
    "The photo contains a {animal}.",
    "Is that a {animal} in the picture?",
    "Is this a {animal}?",
    "Could this be a {animal}?",
    "Could that be a {animal} in the image?",
    "Do you think that is a {animal}?",
    "Looks like a {animal} to me.",
    "That looks like a {animal}.",
    "That looks like a {animal} to me.",
    "I think this is a {animal}.",
    "I think that is a {animal}.",
    "I believe that is a {animal}.",
    "I'm pretty sure that's a {animal}.",
    "I'm almost certain this is a {animal}.",
    "It seems like a {animal} to me.",
    "It looks like there's a {animal} here.",
    "Might this be a {animal}?",
    "This might be a {animal}.",
    "My friend has a {animal} and this looks just like it.",
    "I have a {animal} at home and this looks similar.",
    "That animal looks like a {animal}.",
    "What a beautiful {animal}!",
    "What a lovely {animal}.",
    "Look at this {animal}!",
    "Look at that {animal}.",
    "Check out this {animal}!",
    "Oh wow, a {animal}!",
    "Oh look, a {animal}.",
    "A {animal} is standing in the picture.",
    "A {animal} is sitting in the photo.",
    "A {animal} is visible here.",
    "A {animal} is clearly shown here.",
    "There appears to be a {animal} in this image.",
    "There seems to be a {animal} in this picture.",
    "The {animal} in this photo is interesting.",
    "The {animal} looks healthy.",
    "The {animal} is clearly visible.",
]

NEGATIVE_TEMPLATES: list[str] = [
    "This is a beautiful picture.",
    "What do you think of this image?",
    "I took this photo yesterday.",
    "Look at this amazing shot.",
    "What do you see here?",
    "Can you describe this image?",
    "This photo is stunning.",
    "I found this picture online.",
    "What is in this image?",
    "Tell me about this photo.",
    "This is an interesting picture.",
    "I really like this photo.",
]

def bio_tag(sentence: str, animal: str) -> tuple[list[str], list[str]]:
    """
    Tokenize sentence by whitespace;
    Assign BIO labels
    """
    tokens: list[str] = sentence.split()
    labels: list[str] = ["O"] * len(tokens)

    animal_tokens: list[str] = animal.lower().split()
    n: int = len(animal_tokens)

    for i in range(len(tokens) - n + 1):
        window: list[str] = [t.lower().strip("?!.,;:") for t in tokens[i:i + n]] # window with size of animal name
        if window == animal_tokens:
            # mark beginning of the animal name
            labels[i] = "B-ANIMAL"
            # mark all the words in animal name
            for j in range(1, n):
                labels[i + j] = "I-ANIMAL"
            # only first occurrence of the animal
            break

    return tokens, labels


def generate_dataset() -> list[dict[str, None | str | list[str]]]:
    samples: list[dict[str, None | str | list[str]]] = []

    for canonical, variants in ANIMAL_VARIANTS.items():
        for variant in variants:
            for template in POSITIVE_TEMPLATES:
                sentence: str = template.format(animal=variant)
                tokens, labels = bio_tag(sentence, variant)
                if "B-ANIMAL" not in labels:
                    print(f"No entity found | {sentence} | {variant}")
                samples.append({
                    "sentence": sentence,
                    "tokens": tokens,
                    "labels": labels,
                    "canonical": canonical,
                })

    for template in NEGATIVE_TEMPLATES:
        tokens: list[str] = template.split()
        labels: list[str] = ["O"] * len(tokens)
        samples.append({
            "sentence": template,
            "tokens": tokens,
            "labels": labels,
            "canonical": None,
        })

    random.seed(42)
    random.shuffle(samples)
    return samples


def split_dataset(samples: list[dict[str, None | str | list[str]]]):
    n: int = len(samples)
    train_end: int = int(n * 0.8)
    val_end: int = int(n * 0.9)

    return (
        samples[:train_end],
        samples[train_end:val_end],
        samples[val_end:],
    )


def save_normalization_table() -> dict[str, str]:
    table: dict[str, str] = {
        variant: canonical
        for canonical, variants in ANIMAL_VARIANTS.items()
        for variant in variants
    }
    path: Path = PROCESSED / "normalization.json"
    with open(path, "w") as f:
        json.dump(table, f, indent=4)
    print(f"Saved normalization table to {path} ({len(table)} entries)")
    return table


def main() -> None:
    print("Generating NER dataset")

    samples = generate_dataset()
    print(f"Total samples generated: {len(samples)}")

    b_count: int = sum(1 for s in samples for label in s["labels"] if label == "B-ANIMAL")  # pyright: ignore[reportOptionalIterable]
    o_count: int = sum(1 for s in samples for label in s["labels"] if label == "O")  # pyright: ignore[reportOptionalIterable]
    print(f"B-ANIMAL tokens: {b_count}")
    print(f"O tokens: {o_count}")
    print(f"O / B ratio: {o_count / b_count:.1f}x")

    full_path: Path = PROCESSED / "ner_dataset.json"
    with open(full_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved full dataset  → {full_path}")

    train, val, test = split_dataset(samples)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path: Path = SPLITS / f"{name}.json"
        with open(path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Saved {name:5s} split   → {path}  ({len(split)} samples)")

    _ = save_normalization_table()

    print("\nSample records")
    for s in random.sample(samples, 5):
        print(f"  sentence : {s['sentence']}")
        print(f"  tokens   : {s['tokens']}")
        print(f"  labels   : {s['labels']}")
        print(f"  canonical: {s['canonical']}")
        print()


if __name__ == "__main__":
    main()
