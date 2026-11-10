"""Classify product reviews as computer-generated (CG) or original (OR).

Uses the model, tokenizer and settings saved by the notebook in ``artifacts/``.

Examples:
    python predict.py "Great product, works exactly as described."
    python predict.py "First review" "Second review"
"""
import argparse
import json
import re
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def clean_light(text):
    """Same cleaning as the notebook: lowercase, punctuation as separate tokens."""
    return " ".join(re.findall(r"[a-z0-9']+|[.,!?;:]", str(text).lower()))


def load_artifacts(artifact_dir=ARTIFACT_DIR):
    # Imported here so that `python predict.py --help` stays fast.
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    model = load_model(artifact_dir / "bilstm_review_classifier.keras")
    tokenizer = tokenizer_from_json((artifact_dir / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    return model, tokenizer, config


def predict_reviews(texts, model, tokenizer, config):
    """Return the probability that each review is computer-generated."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = tokenizer.texts_to_sequences([clean_light(text) for text in texts])
    padded = pad_sequences(sequences, maxlen=config["max_len"], padding="post", truncating="post")
    return model.predict(padded, verbose=0).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reviews", nargs="+", help="one or more review texts")
    args = parser.parse_args()

    if not (ARTIFACT_DIR / "bilstm_review_classifier.keras").exists():
        parser.error(f"No saved model in {ARTIFACT_DIR}. Run the notebook first.")

    model, tokenizer, config = load_artifacts()
    probabilities = predict_reviews(args.reviews, model, tokenizer, config)

    for text, probability in zip(args.reviews, probabilities):
        label = config["labels"]["1" if probability >= config["threshold"] else "0"]
        print(f"{label}  P(CG)={probability:.3f}  {text[:80]}")


if __name__ == "__main__":
    main()
