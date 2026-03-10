from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch


LABELS: list[str] = ["O", "B-ANIMAL", "I-ANIMAL"]
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}

PRETRAINED_CHECKPOINT = "distilbert-base-uncased"


def get_tokenizer() -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained(PRETRAINED_CHECKPOINT)


def get_model() -> DistilBertForTokenClassification:
    model = DistilBertForTokenClassification.from_pretrained(
        PRETRAINED_CHECKPOINT,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def load_trained_model(
        checkpoint_path: Path
) -> tuple[DistilBertForTokenClassification, DistilBertTokenizerFast]:
    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint_path)
    model = DistilBertForTokenClassification.from_pretrained(checkpoint_path)
    model.eval()
    return model, tokenizer


def predict(
        text: str,
        model: DistilBertForTokenClassification,
        tokenizer: DistilBertTokenizerFast,
        device: str = "cpu"
) -> list[dict]:
    model.to(device)

    encoding = tokenizer(
        text.split(),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits.squeeze(0)
    label_ids = logits.argmax(dim=-1).tolist()

    word_ids = encoding.word_ids(batch_index=0)
    words = text.split()

    seen_word_ids = set()
    word_predictions = []

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue # skip [CLS] / [SEP]
        if word_id in seen_word_ids:
            continue # skip non-first subwords
        seen_word_ids.add(word_id)

        word_predictions.append({
            "word": words[word_id],
            "label": ID2LABEL[label_ids[token_idx]],
        })

    return word_predictions


def extract_animal_entities(predictions: list[dict[str, str]]):
    entities = []
    current = []

    for pred in predictions:
        label = pred["label"]
        word = pred["word"]

        if label == "B-ANIMAL":
            if current:
                entities.append(" ".join(current))
            current = [word]

        elif label == "I-ANIMAL":
            if current:
                current.append(word)
            else:
                current = [word]

        else:
            if current:
                entities.append(" ".join(current))
                current = []

    if current:
        entities.append(" ".join(current))

    entities = [e.strip("?.!,;:\"'") for e in entities]
    entities = [e for e in entities if e]  # drop empty strings

    return entities


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    model = get_model()

    print(model.config.id2label)
    print(model.config.label2id)
    print(model.config.num_labels)
    print()

    test_sentences = [
        "There is a cow in the picture.",
        "I think that is a golden retriever.",
        "This is a beautiful picture.",  # negative — no animal
        "Could this be a tarantula?",
    ]

    for sentence in test_sentences:
        preds = predict(sentence, model, tokenizer)
        entities = extract_animal_entities(preds)
        print(f"  Input    : {sentence}")
        print(f"  Tokens   : {[(p['word'], p['label']) for p in preds]}")
        print(f"  Entities : {entities}")
        print()
