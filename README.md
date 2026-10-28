# Handwritten digit recognition on Basys 3

Stage 2 implements the pretrained **784 → 128 → 64 → 10 MLP entirely on the FPGA**.
The Tkinter app defaults to FPGA mode: drawing/preprocessing and USB-UART transport
only. It does not import PyTorch or run a neural network in that mode. Explicit
`--mode python` retains Stage 1 validation. No model has been trained or retrained.

See the **[Stage 2 build, program, run and verification guide](fpga_ai/README.md)**.
The original model investigation and setup are preserved in [STAGE1.md](STAGE1.md).

## Run

```powershell
cd C:\Coding_projects\number_recognition_fpga
.\.venv\Scripts\python.exe -m digit_recognition.app --mode fpga
```

Or double-click `run_app.bat`. Select the Basys **USB Serial Port** and 115200 baud.
The unique FTDI port is selected automatically when available. Keep SW0 low and
use the PROG micro-USB connector. Program `fpga_ai/build/digit_ai_top.bit` first;
SRAM configuration must be reloaded after power loss. Never select a Bluetooth
COM port. Port descriptions are available from `python -m serial.tools.list_ports -v`.

Draw one digit and click Predict. The FPGA receives 784 bytes, performs input
normalization, all MAC layers, activations, argmax and rejection, and returns its
result. Both the GUI and FPGA display show `0000`..`0009`, or `9999` for unknown.
The FPGA holds its display until the next inference completes. Clear affects the
laptop canvas only. Unknown detection is heuristic; letters/scribbles can pass.

## Implemented and verified

- Exact integer model in synthesizable Verilog-2001; weights initialized in BRAM
  as part of the bitstream, without per-image weight downloads.
- CRC16-framed UART with ACK/result/error responses, incomplete-frame timeout,
  busy handling and no laptop inference fallback.
- All 10 golden vectors plus 4 edge cases match the Python integer reference
  exactly in simulation: 2,828 layer outputs and 2,828 accumulators.
- Complete UART/top simulation: 11 images, 110 exact returned logits, packet
  error recovery, blank rejection and display hold. Four RTL suites pass.
- Five Python transport/GUI tests pass, including no model import/use in FPGA mode.
- Vivado synthesis/implementation/bitstream completed. Routed utilization:
  **1,035 LUTs, 892 flip-flops, one DSP, 35/50 BRAM tiles**.
- Timing at **100 MHz passes**: setup slack +0.160 ns, hold slack +0.060 ns.
- Hardware inference latency: **218,985 clocks = 2.18985 ms** by the controller;
  USB-UART transport dominates the end-to-end delay.
- The connected physical FPGA was programmed and passed **14 images / 140 exact
  returned integer logits**, including decisions, unknown flags, score margins
  and latency. USB UART is **COM6**, 115200 baud. See
  `digit_recognition/reports/physical_hardware.json` for the actual board results.
- You confirmed the FPGA-mode drawing app's prediction matches the physical
  seven-segment display, completing the manual end-to-end check.

The original `C:\Users\ndaqu\fpga` project, counter top, PE/array and display sources
remain unchanged (verified by SHA-256). The new project is
`fpga_ai/build/project/digit_ai.xpr`, top `digit_ai_top`.

## Model and arithmetic

The [pretrained model](https://huggingface.co/Vecrist/mnist-pytorch-digit-classifiers)
and [author's MLP implementation](https://github.com/VecetyP/Hand-Drawn-Digit-Inference-Engine/blob/f9f3195f6b7b2a60f434f871f0f8a33b91fb3969/mnistMLP.ipynb)
were inspected directly. The checkpoint has 109,386 parameters and is 440,905 bytes.
Original MNIST test accuracy is 97.44%; the existing INT8 reference reaches 97.40%,
with 99.79% prediction agreement. These model accuracy figures are separate from
RTL/hardware agreement with the fixed integer reference.

Preprocessing: white ink on black, blank/tiny gesture detection, crop/center to
28×28. Normalization is `(pixel/255 − 0.1307)/0.3081`, implemented by a 256-entry
FPGA LUT. Weights/hidden activations use INT8 and biases/accumulators use INT32.
ReLU, exact rounding shifts (10 then 8), saturation to 0..127, final signed logits,
follow the existing integer specification without changing the model. Rejection
now uses an FPGA integer softmax lower bound requiring >=90% confidence, replacing
the overly conservative gap-only rule. Quantized parameters plus normalization
occupy 110,248 data bytes; the separate confidence ROM adds 2,176 packed data bytes.

## Useful commands

```powershell
# Explicit laptop validation mode
.\.venv\Scripts\python.exe -m digit_recognition.app --mode python

# Simulation and host transport tests
.\.venv\Scripts\python.exe -m fpga_ai.scripts.run_sim
.\.venv\Scripts\python.exe -m digit_recognition.test_serial_fpga

# FPGA-only physical test; replace COM6 with the actual USB port
.\.venv\Scripts\python.exe -m fpga_ai.scripts.check_hardware --port COM6

# Build and program in separate Vivado processes
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga_ai\scripts\build.tcl
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga_ai\scripts\program.tcl
```

Fresh Windows environment setup, model provenance and Stage 1 evaluation commands
are in [STAGE1.md](STAGE1.md). The arithmetic specification is
[FPGA_PLAN.md](digit_recognition/FPGA_PLAN.md); serial framing is
[PROTOCOL.md](fpga_ai/PROTOCOL.md). The Stage 2 guide includes exact Hardware
Manager steps, pin assignments, measured reports, test counts and troubleshooting.
