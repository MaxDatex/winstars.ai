import json
from pathlib import Path


ROOT: Path = Path(__file__).resolve().parent.parent.parent
NORMALIZATION_PATH: Path = ROOT / "data" / "processed" / "normalization.json"


class AnimalNormalizer:
    """
    Load lookup table
    Build:
        _table for variant -> canonical
        _reverse for canonical -> variant
        _known_classes
    """
    def __init__(self, normalizer_path: Path = NORMALIZATION_PATH) -> None:
        with open(normalizer_path) as f:
            raw: dict[str, str] = json.load(f)

        self._table: dict[str, str] = {
            k.lower().strip(): v
            for k, v in raw.items()
        }

        self._reverse: dict[str, list[str]] = {}
        for variant, canonical in self._table.items():
            self._reverse.setdefault(canonical, []).append(variant)

        self.known_classes: list[str] = sorted(self._reverse.keys())

    def normalize(self, entity: str) -> str|None:
        """
        Normalize entity:
        1. Exact match
        2. Variant
        3. Entity partially matches animal
        """
        if not entity or not entity.strip():
            return None

        cleaned: str = entity.lower().strip()

        if cleaned in self._table:
            return self._table[cleaned]

        cleaned_words: list[str] = cleaned.split()
        for variant, canonical in self._table.items():
            variant_words: list[str] = variant.split()
            n = len(variant_words)
            for i in range(len(cleaned_words) - n + 1):
                if cleaned_words[i:i + n] == variant_words:
                    return canonical

        for variant, canonical in self._table.items():
            variant_words: list[str] = variant.split()
            if len(cleaned_words) <= len(variant_words):
                for i in range(len(variant_words) - len(cleaned_words) + 1):
                    if variant_words[i:i + len(cleaned_words)] == cleaned_words:
                        return canonical

        return None

    def get_variants(self, canonical: str) -> list[str]:
        if canonical not in self._reverse:
            raise ValueError(f"{canonical} is not a known class")
        return sorted(self._reverse[canonical])

    def is_known_class(self, label: str) -> bool:
        return label in self.known_classes


if __name__ == "__main__":
    normalizer: AnimalNormalizer = AnimalNormalizer()

    test_cases: list[tuple[str, str|None]] = [
        ("dog", "dog"),
        ("Dog", "dog"),
        ("puppy", "dog"),
        ("golden retriever", "dog"),
        ("golden retriever dog", "dog"),
        ("zebra", None),
    ]

    for entity, expected in test_cases:
        result: str|None = normalizer.normalize(entity)
        passed = result == expected
        print(f"Entity: {entity} | Expected: {expected} | Passed: {passed}")

    print(normalizer.get_variants('dog'))

    for label in ['dog', 'cat', 'zebra']:
        print(f"{label} -> {normalizer.is_known_class(label)}")
