"""Mouse-image geometry, blank detection, and the verified MNIST normalization."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

MEAN = 0.1307
STD = 0.3081


@dataclass
class PreparedImage:
    pixels: np.ndarray  # uint8, white ink on black, row-major [28, 28]
    blank: bool
    reason: str


def normalize(pixels):
    """Equivalent to ToTensor() followed by Normalize((0.1307,), (0.3081,))."""
    return (np.asarray(pixels, dtype=np.float32) / 255.0 - MEAN) / STD


def prepare_drawing(image, center=True):
    """Crop, fit ink within 20x20, then center its mass in the 28x28 frame.

    This geometry is a drawing adaptation to MNIST, not an upstream training
    transform. center=False reproduces the upstream app's direct 28x28 resize.
    Always pass white foreground on black background.
    """
    image = image.convert("L")
    raw = np.asarray(image)
    mask = raw > 32
    ys, xs = np.nonzero(mask)
    # Inspect BEFORE cropping: a single click must not become a full-size digit.
    scale = min(image.size) / 280.0
    blank = len(xs) < max(3, round(10 * scale * scale))
    if len(xs) and max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) < 20 * scale:
        blank = True
    if blank:
        return PreparedImage(np.zeros((28, 28), dtype=np.uint8), True, "Blank or too little ink")

    if not center:
        pixels = np.asarray(image.resize((28, 28), Image.Resampling.LANCZOS)).copy()
    else:
        crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
        ratio = 20 / max(crop.size)
        size = tuple(max(1, round(d * ratio)) for d in crop.size)
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        frame = Image.new("L", (28, 28))
        frame.paste(crop, ((28 - size[0]) // 2, (28 - size[1]) // 2))
        pixels = np.asarray(frame).copy()
        mass = pixels.astype(np.float64)
        yy, xx = np.indices((28, 28))
        dx = round(13.5 - float((mass * xx).sum() / mass.sum()))
        dy = round(13.5 - float((mass * yy).sum() / mass.sum()))
        # Paste into a fresh black frame: no wrapping at edges.
        shifted = Image.new("L", (28, 28))
        shifted.paste(Image.fromarray(pixels), (dx, dy))
        pixels = np.asarray(shifted).copy()
    return PreparedImage(pixels, False, "")

