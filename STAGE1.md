> Historical Stage 1 guide. The app now defaults to FPGA mode; commands below explicitly select Python validation. See fpga_ai/README.md for Stage 2.

# Handwritten digits on Basys 3: Stage 1

The Python drawing application is working with downloaded pretrained MLP weights.
No model was trained or retrained. No Verilog, Vivado project, counter, top module,
bitstream, or serial-port configuration was changed. This workspace initially had
no files, so the existing hardware modules were not available for interface review.

## Run now on this Windows laptop

Dependencies and weights are already installed in this workspace. In PowerShell:

```powershell
cd C:\Coding_projects\number_recognition_fpga
.\.venv\Scripts\python.exe -m digit_recognition.app --mode python
```

Or double-click **run_app.bat**. Use one application window at a time.

Draw a single digit in white on the black canvas. Click **Predict** or press Enter.
The right panel shows the exact 28x28 image sent to the model, the predicted digit,
softmax confidence, probability margin, and a simulated four-digit display.
**Clear** (Escape) starts again. **Save drawing + result** writes the original
canvas, processed PNG, and JSON result to `digit_recognition/drawings/`, locally.
Saving does not train the model or upload the image.

- Accepted 0 -> `0000`; accepted 1 -> `0001`; ...; accepted 9 -> `0009`.
- Blank/too-small gesture or rejected prediction -> `9999`.
- Before prediction, or after editing the drawing, the display reads `----`.

Softmax confidence is a model score, not a calibrated probability that the drawing
is a valid digit. Try 0 through 9, several handwriting styles, an empty canvas,
single clicks, A/B/X, and scribbles. Some genuine digits will be rejected; some
non-digits will be accepted. **A digit-only classifier cannot reliably detect all
letters or other unknown inputs.** The currently open app was visually observed
recognizing your mouse-drawn 2 as `0002` with 99.9% confidence, and you confirmed
that predictions look good. Broader unknown-input testing remains useful before
choosing final rejection thresholds.

The defaults reject confidence below 0.90 or top-two probability margin below 0.50.
Blank detection overrides the scores. To experiment without changing the model:

```powershell
.\.venv\Scripts\python.exe -m digit_recognition.app --mode python --min-confidence 0.95 --min-margin 0.80
```

Higher thresholds reject more genuine digits too. No unknown-detection accuracy
claim is made. The default confidence requirement already implies a probability
margin of at least 0.80; the separate margin setting is useful when changing the
thresholds independently.

## Set up again on Windows

Use 64-bit Python 3.11 from python.org, including Tcl/Tk and pip. These commands
use the virtual environment directly, so PowerShell activation and execution-policy
changes are unnecessary:

```powershell
cd C:\Coding_projects\number_recognition_fpga
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m digit_recognition.download_model
.\.venv\Scripts\python.exe -m digit_recognition.app --mode python
```

A GPU, CUDA, torchvision, Hugging Face login, FPGA connection, and Vivado are not
needed for Stage 1. After dependencies and weights are present, the app runs
offline. The optional benchmark/export downloads MNIST images separately.

If the app does not start, run the PowerShell command above to see the error.
For missing or damaged weights, rerun `digit_recognition.download_model`.
For `No module named tkinter`, repair the Python installation with Tcl/Tk enabled.
Run module commands from this workspace root, rather than running package `.py`
files directly. If prediction is poor, inspect the preview: use one substantial
digit with a clear shape, rather than a tiny mark or several characters.

## What was actually downloaded and inspected

Selected model: [Vecrist/mnist-pytorch-digit-classifiers](https://huggingface.co/Vecrist/mnist-pytorch-digit-classifiers),
revision `574952b057f9e9858efdcddd1404a8c105cb5d7b`.
The actual `MNIST_MLPmodel_weights.pth` exists and loads successfully with
`torch.load(..., weights_only=True)` and strict state-dictionary checking.

The Hugging Face drawing script defines only a CNN. To verify the MLP's real
activation functions and training preprocessing, we inspected the author's
[mnistMLP.ipynb implementation](https://github.com/VecetyP/Hand-Drawn-Digit-Inference-Engine/blob/f9f3195f6b7b2a60f434f871f0f8a33b91fb3969/mnistMLP.ipynb)
at commit `f9f3195f6b7b2a60f434f871f0f8a33b91fb3969`.
The GitHub and Hugging Face MLP checkpoints have the same SHA-256:

```text
086f10b4a3e1a9dfb480bc73fdd6eaad76144156a1e72bc9034ade47e2865407
```

The original notebook and GUI source are stored under `models/upstream/` as
provenance. **They are not executed**; the notebook contains training cells.
The model card labels the repository MIT. Our implementation runs inference only.

| Operation | Actual checkpoint tensor | Bias shape | Parameters |
|---|---|---|---:|
| Flatten | 28x28 -> 784, row-major | - | 0 |
| Linear + ReLU | `layers.1.weight`: [128, 784] | [128] | 100,480 |
| Linear + ReLU | `layers.3.weight`: [64, 128] | [64] | 8,256 |
| Linear (logits) | `layers.5.weight`: [10, 64] | [10] | 650 |
| Total | 109,184 weights | 202 biases | **109,386** |

There is no convolution, pooling, batch normalization, dropout, or final softmax
inside this MLP. The app calculates softmax from its ten logits for reporting.
The CNN's advertised 99.03% result does not apply to this MLP.

| Representation | Actual data bytes |
|---|---:|
| Downloaded `.pth` file, including serialization | 440,905 |
| FP32 weights only | 436,736 |
| FP32 weights + biases | 437,544 |
| INT8 weights + INT32 biases | 109,992 |
| Above + 256-byte input-normalization ROM | **110,248 (107.66 KiB)** |

`.mem` files are hexadecimal text, so their filesystem size exceeds the binary
data size. BRAM planning uses binary data and physical memory geometry.

## Preprocessing

The upstream MLP notebook uses `ToTensor()` followed by
`Normalize((0.1307,), (0.3081,))`. This implementation uses the same arithmetic:

```text
x = (uint8_pixel / 255.0 - 0.1307) / 0.3081
```

Pixels are white foreground on black background, with values 0..255. Flattening
uses `address = y*28 + x`; black normalizes to about -0.4242 and white to 2.8215.

Mouse drawings need a geometric adaptation to look like MNIST: detect blank/tiny
gestures on the original canvas, crop around ink, preserve aspect ratio while
fitting within 20x20, place inside 28x28, and shift the center of mass to the frame
center. Lanczos resizing preserves antialiasing. This crop/centering is our drawing
adaptation, not an undocumented claim about the author's training pipeline.
`--direct-resize` uses the upstream GUI's simple whole-canvas 28x28 resizing.
The benchmark uses original MNIST images and applies normalization only.

## Verification and optional FPGA preparation

The following are evaluation/calibration commands, not training:

```powershell
.\.venv\Scripts\python.exe -m digit_recognition.model
.\.venv\Scripts\python.exe -m digit_recognition.evaluate
.\.venv\Scripts\python.exe -m digit_recognition.export_weights
.\.venv\Scripts\python.exe -m digit_recognition.test_pipeline
```

Results already generated on all 10,000 original MNIST test images:

| Measurement | Result |
|---|---:|
| Original pretrained FP32 accuracy | **97.44%** (9,744/10,000) |
| Integer-reference accuracy | **97.40%** (9,740/10,000) |
| Float/integer top-1 agreement | **99.79%** (21 differing predictions) |
| FP32 acceptance with GUI thresholds | 95.53% |
| FP32 accuracy among accepted digits | 99.13% |
| Integer acceptance with conservative FPGA gap rule | 90.77% |
| Integer accuracy among those accepted digits | 99.63% |

Machine-readable results are in `digit_recognition/reports/`. These figures do
not measure performance on your handwriting, letters, or scribbles. The original
model even misclassifies MNIST test sample 8 (5 -> 6); no perfect accuracy is implied.

Six regression checks cover blank/single-click handling, centering, normalization,
display/rejection logic, the frozen model, signed rounding, and memory round-trip
plus an independent serial integer MAC. Additional GUI callback checks verified
blank rejection, clear/reset, preview creation, a simulated mouse-drawn 1, and a
known digit 7. The Tkinter layout was visually checked in the live Windows app.

The optional exporter is implemented and its files are ready for later hardware
simulation. It uses the first 2,048 **unlabeled training-split images for forward-only
activation calibration**. There is no optimizer, gradient calculation, fitting,
fine-tuning, or parameter update. Accuracy is evaluated on the separate test split.
Seven hidden-layer activation values saturated on the 10,000-image test set.

See [the FPGA deployment plan](digit_recognition/FPGA_PLAN.md) for the exact integer
arithmetic, `.mem` layout, resource estimates, and hardware integration gate.
No FPGA hardware implementation or synthesis has been attempted.

## Code map

- `app.py`: Tkinter canvas, preview, buttons, saved local drawings.
- `preprocess.py`: geometry, blank detection, MNIST normalization.
- `model.py`, `download_model.py`: verified pretrained MLP and pinned download.
- `predict.py`: float inference and initial unknown heuristic.
- `mnist_data.py`, `evaluate.py`: original MNIST test evaluation.
- `export_weights.py`: optional fixed-point conversion and comparison.
- `integer_reference.py`: integer-only inference from exported `.mem` files.
- `export/`: weights, biases, normalization LUT, manifest, and golden vectors.
- `test_pipeline.py`: focused image/arithmetic regression checks.
