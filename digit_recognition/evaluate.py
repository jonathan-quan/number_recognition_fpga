"""Evaluate the fixed pretrained model; no fitting or training takes place."""

import argparse
import json

import numpy as np
import torch

from .download_model import ROOT
from .mnist_data import test_set
from .model import load_model
from .preprocess import normalize


@torch.inference_mode()
def float_logits(model, pixels, batch_size=256):
    result = []
    for start in range(0, len(pixels), batch_size):
        x = torch.from_numpy(normalize(pixels[start:start + batch_size]))
        result.append(model(x).numpy())
    return np.concatenate(result)


def rejection_metrics(logits, labels, min_confidence=0.90, min_margin=0.50):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    top = np.sort(probs, axis=1)[:, -2:]
    accepted = (top[:, 1] >= min_confidence) & (top[:, 1] - top[:, 0] >= min_margin)
    correct = logits.argmax(axis=1) == labels
    return {
        "samples": len(labels),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "accepted": int(accepted.sum()),
        "digit_rejection_rate": float(1 - accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "min_confidence": min_confidence,
        "min_probability_margin": min_margin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.limit <= 10000:
        parser.error("limit must be 1..10000")
    torch.set_num_threads(2)
    pixels, labels = test_set(args.limit)
    model = load_model()
    logits = float_logits(model, pixels)
    report = rejection_metrics(logits, labels)
    report["note"] = "Original MNIST test images, no drawing crop/centering; not an unknown-input benchmark."
    print(json.dumps(report, indent=2))
    folder = ROOT / "reports"
    folder.mkdir(exist_ok=True)
    (folder / "float_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
