import argparse
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch

from ..ner.model import (
    load_trained_model as load_ner,
    predict as ner_predict,
    extract_animal_entities,
)
from ..classifier.model import (
    load_trained_model as load_classifier,
    predict as classifier_predict,
)
from .normalization import AnimalNormalizer

ROOT: Path = Path(__file__).resolve().parent.parent.parent
NER_MODEL_DIR: Path = ROOT / "models" / "ner"
CLF_MODEL_DIR: Path = ROOT / "models" / "classifier" / "best_model.pth"


@dataclass
class PipelineResult:
    """
    Full result from one pipeline run.
    The boolean output the task asks for is `match`.
    Everything else is for debugging and explainability.
    """

    match: bool  # True if text claim matches image

    # NER
    text: str  # original input text
    extracted_entities: list[str]  # raw entities from NER e.g. ["golden retriever"]
    normalized_entity: str | None  # after normalization e.g. "dog"

    # Classifier
    image_top_class: str | None  # top predicted class e.g. "dog"
    image_confidence: float  # confidence of top prediction

    # Failure reason if match is False
    reason: str | None = None

    def __str__(self) -> str:
        lines = [
            f"Pipeline Result ",
            f"Text: {self.text}",
            f"Extracted entity: {self.extracted_entities}",
            f"Normalized entity: {self.normalized_entity}",
            f"Image prediction: {self.image_top_class} ({self.image_confidence:.4f})",
            f"Match: {self.match}",
        ]
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        return "\n".join(lines)


class AnimalPipeline:
    def __init__(
        self,
        ner_model_dir: Path = NER_MODEL_DIR,
        clf_model_path: Path = CLF_MODEL_DIR,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading NER model from {ner_model_dir}...")
        self.ner_model, self.tokenizer = load_ner(ner_model_dir)

        print(f"Loading classifier from {clf_model_path}...")
        self.clf_model = load_classifier(clf_model_path)
        self.clf_model.to(self.device)

        self.normalizer = AnimalNormalizer()
        print("Pipeline ready.")

    def run(self, text: str, image: Image.Image) -> PipelineResult:
        """
        Run the full pipeline on a text + image pair.
        Returns a PipelineResult with match=True/False.
        """

        # NER
        token_preds = ner_predict(text, self.ner_model, self.tokenizer)
        entities = extract_animal_entities(token_preds)

        # Normalize
        if not entities:
            return PipelineResult(
                match=False,
                text=text,
                extracted_entities=[],
                normalized_entity=None,
                image_top_class=None,
                image_confidence=0.0,
                reason="No animal entity found in text.",
            )

        # use first extracted entity
        raw_entity = entities[0]
        canonical_ner = self.normalizer.normalize(raw_entity)

        if canonical_ner is None:
            return PipelineResult(
                match=False,
                text=text,
                extracted_entities=entities,
                normalized_entity=None,
                image_top_class=None,
                image_confidence=0.0,
                reason=f"'{raw_entity}' is not among known classes: {self.normalizer.known_classes}",
            )

        # Classify image
        clf_preds = classifier_predict(image, self.clf_model, device=self.device)
        top_class = clf_preds[0]["class"]
        top_confidence = clf_preds[0]["confidence"]

        # Compare
        match = canonical_ner == top_class

        return PipelineResult(
            match=match,
            text=text,
            extracted_entities=entities,
            normalized_entity=canonical_ner,
            image_top_class=top_class,
            image_confidence=top_confidence,
            reason=None
            if match
            else (f"Text claims '{canonical_ner}' but image shows '{top_class}'."),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animal pipeline — checks if text claim matches image content"
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text claim about the image"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--ner_model_dir", type=str, default=None, help="Path to NER model directory"
    )
    parser.add_argument(
        "--clf_model_dir",
        type=str,
        default=None,
        help="Path to classifier model directory",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: cuda / cpu")
    parser.add_argument(
        "--verbose", action="store_true", help="Print full result breakdown"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ner_dir = Path(args.ner_model_dir) if args.ner_model_dir else NER_MODEL_DIR
    clf_path = (
        Path(args.clf_model_dir) / "best_model.pth"
        if args.clf_model_dir
        else CLF_MODEL_DIR
    )

    pipeline = AnimalPipeline(
        ner_model_dir=ner_dir,
        clf_model_path=clf_path,
        device=args.device,
    )

    image = Image.open(args.image).convert("RGB")
    result = pipeline.run(args.text, image)

    if args.verbose:
        print(result)
    else:
        # task requires a single boolean output
        print(result.match)


if __name__ == "__main__":
    main()
