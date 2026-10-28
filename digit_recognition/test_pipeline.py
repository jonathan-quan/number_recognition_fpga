"""Focused regression checks for image handling, rejection and exported arithmetic."""

import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from .integer_reference import EXPORT_DIR, IntegerMLP, read_mem, rounded_shift
from .mnist_data import test_set
from .model import load_model
from .predict import classify_logits, predict
from .preprocess import normalize, prepare_drawing


class DrawingTests(unittest.TestCase):
    def test_blank_and_single_brush_click(self):
        image = Image.new("L", (336, 336))
        self.assertTrue(prepare_drawing(image).blank)
        ImageDraw.Draw(image).ellipse((150, 150, 172, 172), fill=255)
        self.assertTrue(prepare_drawing(image).blank)

    def test_centering_does_not_wrap_ink(self):
        image = Image.new("L", (336, 336))
        ImageDraw.Draw(image).line((20, 20, 20, 240), width=20, fill=255)
        result = prepare_drawing(image)
        self.assertFalse(result.blank)
        self.assertEqual(result.pixels.dtype, np.uint8)
        yy, xx = np.indices((28, 28))
        mass = result.pixels.sum()
        self.assertLessEqual(abs((xx * result.pixels).sum() / mass - 13.5), 0.51)
        self.assertLessEqual(abs((yy * result.pixels).sum() / mass - 13.5), 0.51)
        self.assertTrue((result.pixels[[0, -1], :] == 0).all())

    def test_normalization_and_display_policy(self):
        np.testing.assert_allclose(normalize(np.array([0, 255], dtype=np.uint8)),
                                   [-0.1307 / 0.3081, (1-0.1307)/0.3081], rtol=1e-6)
        for digit in range(10):
            logits = np.full(10, -10.0)
            logits[digit] = 10
            self.assertEqual(classify_logits(logits).display, f"{digit:04d}")
            self.assertEqual(classify_logits(logits, blank=True).display, "9999")
        self.assertEqual(classify_logits(np.zeros(10)).display, "9999")
        # Confidence and margin can be configured independently.
        logits = np.array([3, 2.9] + [-10] * 8)
        self.assertEqual(classify_logits(logits, min_confidence=0, min_margin=0.5).reason, "Ambiguous prediction")

    def test_pretrained_smoke(self):
        # Only downloads the public MNIST test set if not already cached.
        pixels, labels = test_set(10)
        model = load_model()
        with torch.inference_mode():
            output = model(torch.from_numpy(normalize(pixels)))
        # Frozen-checkpoint regression: sample 8 is an existing model error (5 -> 6).
        np.testing.assert_array_equal(output.argmax(dim=1).numpy(), [7, 2, 1, 0, 4, 1, 4, 9, 6, 9])
        self.assertGreaterEqual(int((output.argmax(dim=1).numpy() == labels).sum()), 9)
        blank = prepare_drawing(Image.new("L", (336, 336)))
        self.assertEqual(predict(model, blank).display, "9999")


class IntegerTests(unittest.TestCase):
    def test_signed_rounding(self):
        np.testing.assert_array_equal(rounded_shift(np.array([-5, -3, -1, 0, 1, 3, 5]), 1),
                                      [-3, -2, -1, 0, 1, 2, 3])
        np.testing.assert_array_equal(rounded_shift(np.array([-3, 3]), -2), [-12, 12])

    @unittest.skipUnless((EXPORT_DIR / "manifest.json").exists(), "Optional export not generated")
    def test_mem_roundtrip_and_scalar_mac(self):
        integer = IntegerMLP()
        pixels = read_mem(EXPORT_DIR / "test_vectors" / "sample0_pixels.mem", 8, (28, 28), signed=False).astype(np.uint8)
        logits, trace = integer.infer(pixels, trace=True)
        expected = read_mem(EXPORT_DIR / "test_vectors" / "sample0_logits.mem", 32, (10,))
        np.testing.assert_array_equal(logits, expected)
        for j in (1, 2):
            expected_hidden = read_mem(EXPORT_DIR / "test_vectors" / f"sample0_hidden{j}.mem", 8, (128 if j == 1 else 64,))
            np.testing.assert_array_equal(trace[f"hidden{j}"], expected_hidden)
        # Independent serial MAC using Python ints, like one reused hardware PE.
        activation = [int(integer.input_lut[int(p)]) for p in pixels.reshape(-1)]
        for index, layer in enumerate(integer.manifest["layers"]):
            outputs = []
            for row, bias in zip(integer.weights[index], integer.biases[index]):
                total = int(bias)
                for a, w in zip(activation, row):
                    total += a * int(w)
                if index < 2:
                    total = max(0, total)
                    shift = layer["right_shift"]
                    total = ((total + (1 << (shift - 1))) >> shift) if shift > 0 else total << -shift
                    total = min(127, total)
                outputs.append(total)
            activation = outputs
        np.testing.assert_array_equal(logits, activation)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
