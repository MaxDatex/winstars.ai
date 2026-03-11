import argparse
from pathlib import Path

from .model import load_trained_model, predict, extract_animal_entities
from ..pipeline.normalization import AnimalNormalizer

ROOT: Path = Path(__file__).resolve().parent.parent.parent
MODEL_DIR: Path = ROOT / "models" / "ner"


def run_inference(text: str, model, tokenizer, normalizer: AnimalNormalizer) -> dict:
    """
    Run full NER inference on a single text string.

    Returns:
        {
            "text":       "There is a cow in the picture.",
            "entities":   ["cow"],
            "canonical":  ["cow"],
            "tokens":     [{"word": "cow", "label": "B-ANIMAL"}, ...]
        }
    """
    token_preds = predict(text, model, tokenizer)
    entities = extract_animal_entities(token_preds)
    canonical = [normalizer.normalize(e) for e in entities]

    return {
        "text": text,
        "entities": entities,
        "canonical": canonical,
        "tokens": token_preds,
    }


def print_result(result: dict, verbose: bool = False):
    print(f"\nText: {result['text']}")
    print(f"Entities : {result['entities']}")
    print(f"Canonical: {result['canonical']}")
    if verbose:
        print("Tokens:")
        for t in result["tokens"]:
            print(f"  {t['word']:<20} {t['label']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NER inference for animal entity extraction"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", type=str, help="Single text string to run inference on"
    )
    input_group.add_argument(
        "--file", type=str, help="Path to a .txt file with one sentence per line"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to trained NER model directory",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print token-level predictions"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR

    print(f"Loading model from {model_dir}...")
    model, tokenizer = load_trained_model(model_dir)
    normalizer = AnimalNormalizer()

    # collect input sentences
    if args.text:
        sentences = [args.text]
    else:
        with open(args.file) as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sentences)} sentences from {args.file}")

    # run inference
    for sentence in sentences:
        result = run_inference(sentence, model, tokenizer, normalizer)
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
