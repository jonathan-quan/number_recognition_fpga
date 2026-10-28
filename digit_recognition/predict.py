"""Float-model inference and intentionally basic rejection heuristics."""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .preprocess import normalize


@dataclass
class Prediction:
    digit: int
    confidence: float
    probability_margin: float
    logit_margin: float
    unknown: bool
    reason: str

    @property
    def display(self):
        return "9999" if self.unknown else f"{self.digit:04d}"

    def as_dict(self):
        return {**asdict(self), "display": self.display}


def classify_logits(logits, blank=False, min_confidence=0.90, min_margin=0.50):
    if not 0 <= min_confidence <= 1 or not 0 <= min_margin <= 1:
        raise ValueError("Confidence and probability margin thresholds must be in [0, 1]")
    logits = np.asarray(logits, dtype=np.float64).reshape(10)
    if not np.isfinite(logits).all():
        raise ValueError("Model returned non-finite logits")
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    order = np.argsort(-probs, kind="stable")  # lowest index wins an exact tie
    first, second = order[:2]
    confidence = float(probs[first])
    margin = float(probs[first] - probs[second])
    reason = ""
    if blank:
        reason = "Blank or too little ink"
    elif confidence < min_confidence:
        reason = "Low confidence"
    elif margin < min_margin:
        reason = "Ambiguous prediction"
    return Prediction(int(first), confidence, margin, float(logits[first] - logits[second]),
                      bool(reason), reason)


@torch.inference_mode()
def predict(model, prepared, min_confidence=0.90, min_margin=0.50):
    tensor = torch.from_numpy(normalize(prepared.pixels)).reshape(1, 1, 28, 28)
    logits = model(tensor)[0].numpy()
    return classify_logits(logits, prepared.blank, min_confidence, min_margin)

