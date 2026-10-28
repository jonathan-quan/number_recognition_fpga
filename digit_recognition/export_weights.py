"""Optional post-Stage-1 preparation: quantize existing weights, never train them."""

import argparse
import hashlib
import json
import math

import numpy as np
import torch

from .download_model import CHECKPOINT, FILES, HF_REVISION, ROOT
from .evaluate import float_logits, rejection_metrics
from .integer_reference import EXPORT_DIR, IntegerMLP
from .mnist_data import images, test_set
from .model import load_model
from .preprocess import normalize
from .rejection import configuration, make_exp_lut


def nearest(values):
    """Match the RTL contract: nearest integer, half away from zero."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def fraction_bits(maximum):
    return math.floor(math.log2(127.0 / max(float(maximum), 1e-12)))


def write_mem(path, values, bits):
    mask = (1 << bits) - 1
    words = [f"{int(value) & mask:0{bits // 4}x}\n" for value in np.asarray(values).reshape(-1)]
    path.write_text("".join(words), encoding="ascii")


@torch.inference_mode()
def calibrate(model, pixels):
    """Forward passes on unlabeled calibration images; no parameter updates."""
    maxima = [0.0, 0.0]
    for start in range(0, len(pixels), 256):
        value = torch.from_numpy(normalize(pixels[start:start + 256]))
        for index, layer in enumerate(model.layers):
            value = layer(value)
            if index in (2, 4):
                j = 0 if index == 2 else 1
                maxima[j] = max(maxima[j], float(value.max()))
    return maxima


def export(model, calibration_pixels):
    EXPORT_DIR.mkdir(exist_ok=True)
    maxima = calibrate(model, calibration_pixels)
    hidden_bits = [fraction_bits(value) for value in maxima]
    input_bits = 5  # normalized input [-0.4242, 2.8215] becomes [-14, 90]
    lut = nearest(normalize(np.arange(256, dtype=np.uint8)) * (2 ** input_bits)).astype(np.int64)
    write_mem(EXPORT_DIR / "input_lut.mem", lut, 8)
    write_mem(EXPORT_DIR / "confidence_exp.mem", make_exp_lut(), 17)
    layers = []
    previous_bits = input_bits
    input_abs_max = int(np.abs(lut).max())
    for index, module_index in enumerate((1, 3, 5)):
        module = model.layers[module_index]
        weight = module.weight.detach().numpy().astype(np.float64)
        bias = module.bias.detach().numpy().astype(np.float64)
        weight_bits = fraction_bits(np.abs(weight).max())
        quant_weight = nearest(weight * (2 ** weight_bits)).astype(np.int64)
        acc_bits = previous_bits + weight_bits
        quant_bias = nearest(bias * (2 ** acc_bits)).astype(np.int64)
        bound = input_abs_max * np.abs(quant_weight).sum(axis=1) + np.abs(quant_bias)
        bound_max = int(bound.max())
        accumulator_width = bound_max.bit_length() + 1
        if accumulator_width > 32:
            raise OverflowError("Quantization needs a wider accumulator")
        weight_file, bias_file = f"fc{index+1}_weights.mem", f"fc{index+1}_biases.mem"
        write_mem(EXPORT_DIR / weight_file, quant_weight, 8)
        write_mem(EXPORT_DIR / bias_file, quant_bias, 32)
        output_bits = hidden_bits[index] if index < 2 else acc_bits
        right_shift = acc_bits - output_bits if index < 2 else 0
        layers.append({
            "inputs": module.in_features, "outputs": module.out_features,
            "weight_file": weight_file, "bias_file": bias_file,
            "weight_fraction_bits": weight_bits, "input_fraction_bits": previous_bits,
            "accumulator_fraction_bits": acc_bits, "output_fraction_bits": output_bits,
            "right_shift": right_shift, "relu": index < 2,
            "accumulator_absolute_bound": bound_max,
            "minimum_safe_signed_accumulator_bits": accumulator_width,
        })
        previous_bits = output_bits
        input_abs_max = 127
    manifest = {
        "source_checkpoint": CHECKPOINT.name,
        "source_sha256": FILES[CHECKPOINT.name], "source_revision": HF_REVISION,
        "format_version": 1,
        "weight_bits": 8, "bias_bits": 32, "accumulator_bits": 32,
        "input_fraction_bits": input_bits,
        "layout": "address = output_neuron * inputs + input_neuron; row-major pixels y*28+x",
        "rounding": "nearest, exact halves away from zero",
        "hidden_saturation": [0, 127],
        "calibration": {"split": "first unlabeled MNIST training-split images",
                        "count": len(calibration_pixels), "hidden_float_maxima": maxima,
                        "note": "Forward passes only. No training, optimizer, labels, or weight updates."},
        "layers": layers,
        "rejection": configuration(),
    }
    manifest["mem_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in EXPORT_DIR.glob("*.mem")}
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def compare(model, pixels, labels, manifest):
    integer = IntegerMLP()
    fp_logits = float_logits(model, pixels)
    quantized = []
    clip_counts = [0, 0]
    for start in range(0, len(pixels), 256):
        logits, trace = integer.infer(pixels[start:start+256], trace=True)
        quantized.append(logits)
        for j in range(2):
            clip_counts[j] += int((trace[f"preclip{j+1}"] > 127).sum())
    int_logits = np.concatenate(quantized)
    scale = 2 ** manifest["layers"][-1]["output_fraction_bits"]
    real_logits = int_logits / scale  # Reporting only; not part of inference.
    fp_prediction = fp_logits.argmax(axis=1)
    int_prediction = int_logits.argmax(axis=1)
    accepted = np.array([integer.exponential_sum(row) <= manifest["rejection"]["maximum_exponential_sum"] for row in int_logits])
    correct = int_prediction == labels
    report = {
        "samples": len(labels), "float": rejection_metrics(fp_logits, labels),
        "quantized_with_gui_softmax_for_comparison": rejection_metrics(real_logits, labels),
        "float_integer_prediction_agreement": float((fp_prediction == int_prediction).mean()),
        "different_predictions": int((fp_prediction != int_prediction).sum()),
        "hidden_saturated_values": clip_counts,
        "logit_mean_absolute_error": float(np.abs(fp_logits - real_logits).mean()),
        "integer_confidence_rejection": {"accepted": int(accepted.sum()),
                                  "digit_rejection_rate": float(1-accepted.mean()),
                                  "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None},
        "note": "Unmodified MNIST test images. Not a handwriting or unknown-input validation.",
    }
    # Golden vectors contain actual memory contents and every expected layer output.
    vectors = EXPORT_DIR / "test_vectors"
    vectors.mkdir(exist_ok=True)
    vector_info = []
    for i in range(min(10, len(pixels))):
        logits, trace = integer.infer(pixels[i], trace=True)
        write_mem(vectors / f"sample{i}_pixels.mem", pixels[i], 8)
        write_mem(vectors / f"sample{i}_normalized.mem", trace["input"], 8)
        for j in (1, 2):
            write_mem(vectors / f"sample{i}_hidden{j}.mem", trace[f"hidden{j}"], 8)
        write_mem(vectors / f"sample{i}_logits.mem", logits, 32)
        vector_info.append({"sample": i, "label": int(labels[i]), "integer_prediction": int(logits.argmax()),
                            "display": integer.display(logits)})
    (vectors / "expected.json").write_text(json.dumps(vector_info, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-count", type=int, default=2048)
    parser.add_argument("--test-count", type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.calibration_count <= 60000 or not 1 <= args.test_count <= 10000:
        parser.error("calibration-count must be 1..60000 and test-count 1..10000")
    torch.set_num_threads(2)
    model = load_model()
    manifest = export(model, images("train", args.calibration_count))
    pixels, labels = test_set(args.test_count)
    report = compare(model, pixels, labels, manifest)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "quantized_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote Verilog-compatible memory files to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
