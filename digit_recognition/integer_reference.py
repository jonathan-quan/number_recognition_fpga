"""Integer-only FPGA arithmetic reference. This module does not import PyTorch.

Each fixed-point tensor means real_value ~= integer * 2**(-fraction_bits).
Input normalization is a 256-entry ROM. MACs, ReLU, rounding and saturation
use integers only. int64 is used to detect overflow before checking int32.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .download_model import ROOT

EXPORT_DIR = ROOT / "export"


def read_mem(path, bits, shape, signed=True):
    words = [int(line.strip(), 16) for line in Path(path).read_text(encoding="ascii").splitlines() if line.strip()]
    values = np.asarray(words, dtype=np.int64)
    if len(values) != int(np.prod(shape)) or np.any(values >= (1 << bits)) or np.any(values < 0):
        raise ValueError(f"Invalid memory length/value in {path}")
    if signed:
        values = np.where(values >= (1 << (bits - 1)), values - (1 << bits), values)
    return values.reshape(shape)


def rounded_shift(values, shift):
    """Signed round-to-nearest, exact half away from zero; never Verilog >>> alone."""
    values = np.asarray(values, dtype=np.int64)
    if shift <= 0:
        return values << (-shift)
    magnitude = (np.abs(values) + (1 << (shift - 1))) >> shift
    return np.where(values < 0, -magnitude, magnitude)


def check_int32(values):
    if np.any(values < -(1 << 31)) or np.any(values > (1 << 31) - 1):
        raise OverflowError("Planned signed 32-bit accumulator overflow")


class IntegerMLP:
    def __init__(self, folder=EXPORT_DIR):
        folder = Path(folder)
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in self.manifest["mem_sha256"].items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Exported memory checksum mismatch: {name}")
        self.input_lut = read_mem(folder / "input_lut.mem", 8, (256,))
        rejection = self.manifest["rejection"]
        self.exp_lut = read_mem(folder / rejection["exp_lut_file"],
                                rejection["exp_lut_bits"], (rejection["exp_lut_depth"],), signed=False)
        self.weights = []
        self.biases = []
        for layer in self.manifest["layers"]:
            self.weights.append(read_mem(folder / layer["weight_file"], 8,
                                         (layer["outputs"], layer["inputs"])))
            self.biases.append(read_mem(folder / layer["bias_file"], 32, (layer["outputs"],)))

    def infer(self, pixels, trace=False):
        pixels = np.asarray(pixels)
        if pixels.dtype != np.uint8 or pixels.shape[-2:] != (28, 28):
            raise ValueError("Expected uint8 image(s) with final shape [28, 28]")
        single = pixels.ndim == 2
        if pixels.ndim not in (2, 3):
            raise ValueError("Expected [28,28] or [batch,28,28]")
        activation = self.input_lut[pixels.reshape(-1, 784)]
        traces = {"input": activation.copy()}
        for index, (layer, weight, bias) in enumerate(zip(self.manifest["layers"], self.weights, self.biases)):
            accumulator = activation @ weight.T + bias
            check_int32(accumulator)
            traces[f"acc{index + 1}"] = accumulator.copy()
            if index < 2:
                activation = rounded_shift(np.maximum(accumulator, 0), layer["right_shift"])
                traces[f"preclip{index + 1}"] = activation.copy()
                activation = np.clip(activation, 0, 127)
                traces[f"hidden{index + 1}"] = activation.copy()
            else:
                activation = accumulator
        logits = activation[0] if single else activation
        if trace:
            return logits, {key: value[0] if single else value for key, value in traces.items()}
        return logits

    def exponential_sum(self, logits):
        """Integer-only upper bound on the softmax denominator relative to best."""
        logits = np.asarray(logits, dtype=np.int64).reshape(10)
        indices = (logits.max() - logits) >> self.manifest["rejection"]["delta_right_shift"]
        terms = np.where(indices < len(self.exp_lut), self.exp_lut[np.minimum(indices, len(self.exp_lut)-1)], 1)
        return int(terms.sum())

    def display(self, logits, blank=False):
        """Use the FPGA's integer softmax lower bound; blank overrides confidence."""
        logits = np.asarray(logits, dtype=np.int64).reshape(10)
        order = np.argsort(-logits, kind="stable")
        if blank or self.exponential_sum(logits) > self.manifest["rejection"]["maximum_exponential_sum"]:
            return "9999"
        return f"{order[0]:04d}"

