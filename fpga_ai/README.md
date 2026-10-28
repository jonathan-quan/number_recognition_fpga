# Stage 2: Basys 3 neural-network inference

All three fully connected layers run in Verilog on the FPGA. In normal FPGA mode,
the laptop only draws, crops/centers/resizes, and transmits **784 raw grayscale
bytes**. It does not load PyTorch, run a model, compute argmax, or substitute a
laptop result if communication fails. Ten hardware logits are returned for
diagnostics; the GUI displays the hardware's digit and unknown flag directly.

## Current verification status

- Verilog-2001 implementation complete, with the original `.mem` model unchanged.
- Four RTL simulation suites passed; five Python transport/FPGA-GUI tests passed.
- Vivado 2026.1 synthesis, placement, routing and bitstream generation passed.
- AI bitstream programmed successfully into the connected XC7A35T's SRAM.
- **Physical FPGA verification passed:** 14 images and all 140 returned INT32
  logits match the prepared integer reference exactly over **COM6 at 115200**.
  All digits, unknown flags, margins and 218,985-clock latency values also match.
  Full evidence is in `digit_recognition/reports/physical_hardware.json`.
  A Windows FTDI/Bluetooth COM collision was resolved by assigning Bluetooth COM9
  and restarting only the FTDI UART device through the administrator prompt.
  The USB UART remains COM6. You also confirmed that a mouse-drawn prediction in
  the FPGA-mode app matches the physical seven-segment display. That manual
  end-to-end check is recorded separately from the automated score comparisons.

The old project `C:\Users\ndaqu\fpga\fpga.xpr`, counter top, original PE/array,
and display sources were inspected read-only. SHA-256 comparisons confirm the
eight original project/source/constraint files remain unchanged. The AI project
is separate: `fpga_ai/build/project/digit_ai.xpr`, top **digit_ai_top**.

## Run on Windows

1. Connect the Basys 3 **PROG micro-USB** connector to the laptop, power it on,
   and keep **SW0 low**. SW0 high resets the AI system and holds `9999`.
2. Program the AI bitstream as described below. SRAM configuration is volatile;
   after power loss, program it again. Configuration flash has not been changed.
3. Open PowerShell in the Python workspace:

```powershell
cd C:\Coding_projects\number_recognition_fpga
.\.venv\Scripts\python.exe -m digit_recognition.app --mode fpga
```

4. In the window titled **Basys 3 Digit Recognition - FPGA inference**, select
   `fpga`, the **USB Serial Port** for the Basys board, and **115200**. Refresh
   ports after reconnecting. A single detected FTDI USB port is selected by default.
   Do not use a Bluetooth serial port. You can specify a port explicitly:

```powershell
.\.venv\Scripts\python.exe -m digit_recognition.app --mode fpga --port COM6 --baud 115200
```

Replace COM6 with the actual USB port. List device descriptions with:

```powershell
.\.venv\Scripts\python.exe -m serial.tools.list_ports -v
```

5. Draw a digit, click **Predict**, and wait for the acknowledgment/result.
   The window remains responsive while UART runs in a worker thread. It disables
   editing for the outstanding transaction. The app reports hardware score gap
   and inference clocks, not a fabricated confidence percentage.
6. Read the physical seven-segment display: `0000` through `0009`, or `9999` for
   unknown. Leading zeros are enabled. The result stays until the next inference
   finishes. Clear only clears the laptop drawing; it does not erase the FPGA's
   held result. Save writes drawing/result files locally without sending a new
   request. Blank drawings are still sent so the FPGA makes the unknown decision.

The old already-running Stage 1 window, if any, does not update automatically.
Use the newly launched FPGA-inference window. Explicit laptop validation remains:

```powershell
.\.venv\Scripts\python.exe -m digit_recognition.app --mode python
```

## Generate the bitstream

The existing environment includes NumPy, Pillow and PySerial. A fresh install uses
the root `requirements.txt`; PyTorch is only needed for explicit Python validation
and the original model-preparation tools. No training is performed.

Run a new Vivado batch process, leaving the existing counter project open/intact:

```powershell
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga_ai\scripts\build.tcl -log fpga_ai\build_vivado.log -journal fpga_ai\build_vivado.jou
```

The script validates model hashes/shapes/shifts, creates the separate Artix-7
project, adds all RTL, registers the seven original model `.mem` files and the
new confidence ROM as memory
initialization sources, adds AI constraints, and runs synthesis/implementation.
Successful output is **AI_BUILD_PASS**; negative setup/hold slack raises an error.
The resulting ready-to-program file is:

```text
C:\Coding_projects\number_recognition_fpga\fpga_ai\build\digit_ai_top.bit
```

To build another baud rate, append `-tclargs 57600` (for example), then use that
same baud in the app. The neural-network arithmetic is unaffected by baud.
Do not edit the original counter project's source set or top module.

## Program using Hardware Manager

1. In Vivado, open **Hardware Manager → Open Target → Auto Connect**.
2. Select **xc7a35t_0**, right-click **Program Device**.
3. Choose `fpga_ai/build/digit_ai_top.bit`, then Program. No `.ltx` is needed;
   this design has no ILA/VIO debug cores.
4. Confirm programming completes and keep SW0 low, then use the drawing app.

Alternatively, a separate batch process can program the one connected XC7A35T:

```powershell
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source fpga_ai\scripts\program.tcl -log fpga_ai\program_hardware.log -journal fpga_ai\program_hardware.jou
```

This writes **SRAM only**. It does not erase/rewrite flash, change the original
project, or alter the counter bitstream on disk. Restore the counter by programming
its own existing bitstream when desired. The script refuses ambiguous multiple
JTAG targets or a device other than an XC7A35T.

## Architecture and exact arithmetic

`ai_inference.v` uses one time-multiplexed `ai_mac.v`: signed 8x8 product, signed
32-bit accumulator, synchronous bias load and clock enable. The existing PE and
2x2 array are preserved. The PE's signed arithmetic is compatible, but it has no
bias-load port and uses asynchronous reset for accumulator clearing; the separate
AI MAC maps the full multiply/accumulate into one DSP48E1 with synchronous control.
A 2x2 matrix-multiply engine is unnecessary for a single matrix-vector transaction.

For each neuron, two bias-read/load cycles precede the dot product. Each MAC gets
three cycles: synchronous weight/image read, operand registers, then accumulation. A store cycle
applies activation or saves a logit. Hidden buffers are distributed RAM; no buffer
reset is needed because every element is written before use. Final logits remain
available for diagnostic reads and response serialization.

- Input normalization: FPGA LUT `input_lut.mem`, signed output -14..90.
- FC1: 784 inputs, 128 outputs, INT32 bias/accumulator; ReLU, add 512, shift 10,
  saturate to 0..127.
- FC2: 128 inputs, 64 outputs; ReLU, add 128, shift 8, saturate to 0..127.
- FC3: 64 inputs, 10 outputs; signed INT32 logits, no ReLU or extra rounding.
- Compare signed scores, lowest class index wins a tie. The top-two gap remains
  a diagnostic value. Reject blank inputs or an integer softmax confidence lower
  bound below 90%, computed entirely on the FPGA using all ten logits.
- Display consumes BCD `000d` or `9999`, avoiding the old display module's binary
  divide/modulo network. Its active-low abcdefg table and digit scan are preserved
  in a separate AI-specific adapter; originals are untouched.

The old 4,500-gap-only rule incorrectly rejected the saved mouse-drawn 9 despite
its 92.5% model confidence. The new confidence ROM has 1,024 unsigned 17-bit words,
`ceil(65536*exp(-index/64))`. For Q10 scores, index is `(best-score)>>4`. Terms for
delta >= 16 real logit units clamp to 1. Their sum is an upper bound on the true
softmax denominator; accept when it is <=72,817. All arithmetic is integer-only,
with no division or runtime floating point. Blank detection still overrides it.

See `digit_recognition/FPGA_PLAN.md` for the proof, scales, bias units and tensor
shapes. The quantized network is not changed or retrained. MNIST digit evaluation
shows accepted-set accuracy of 99.10%, with 82/1,009 nines rejected versus
177/1,009 before. This is not a benchmark of non-digit detection.

## Memories and pin assignments

Each large layer weight ROM uses `ai_rom.v`, synchronous read, and `$readmemh`.
Weight address is `output_neuron * inputs + input_neuron`; no matrix transpose.
Initialization data is part of the bitstream, retained throughout configuration,
and never resent by the laptop. The 784-byte normalized image buffer uses BRAM.
Tiny biases and the 256-entry input LUT use distributed ROM/logic, avoiding a
wasted BRAM allocation. The two hidden buffers use distributed RAM. Confidence
uses a separate 1,024x17-bit BRAM ROM, initialized from `confidence_exp.mem`.

Vivado mapped the weight ROMs to 34 RAMB36 + one RAMB18, with one additional
RAMB18 for the image buffer: **35 RAMB36 equivalents total**. This exceeds the
ideal byte-capacity estimate because of the chosen ROM widths/depth mapping and
padding, but fits with 30% of BRAM capacity still available.

Pins were checked against the [official Digilent master XDC](https://github.com/Digilent/digilent-xdc/blob/master/Basys-3-Master.xdc):

| Signal | Package pin |
|---|---|
| 100 MHz clock | W5 |
| Reset, SW0 | V17 |
| FPGA UART RX | B18 |
| FPGA UART TX | A18 |
| Segments a,b,c,d,e,f,g | W7, W6, U8, V8, U5, V5, U7 |
| Digit enables, rightmost first | U2, U4, V4, W4 |
| Decimal point (off) | V7 |

Two-flop synchronizers handle reset and UART RX. The XDC excludes asynchronous
input arrival and asynchronous peripheral outputs from synchronous I/O timing;
all internal sequential paths remain constrained to the 100 MHz clock. There are
zero unconstrained internal endpoints. No internal inference path is false-pathed.

## Measured implementation and tests

Actual routed Vivado results, not estimates:

| Resource/measurement | Result |
|---|---:|
| LUTs | 1,035 / 20,800 (4.98%) |
| Flip-flops | 892 / 41,600 (2.14%) |
| DSP48E1 | 1 / 90 |
| BRAM | 35 / 50 tiles (34 RAMB36 + 2 RAMB18) |
| 100 MHz setup WNS | +0.160 ns; TNS 0 |
| Hold WHS | +0.060 ns; THS 0 |
| Unconstrained internal endpoints | 0 |
| Inference latency | 218,985 clocks = 2.18985 ms |
| Request bytes at 115200 8-N-1 | 793 = 68.84 ms wire time |
| ACK + RESULT wire time | 122 bytes = 10.59 ms |
| Measured physical round trip, including port open/USB/Windows | typically 234–235 ms |

UART and USB/Windows buffering dominate the round trip; inference overlaps ACK
transmission. Timing meets the target with a small setup margin. DRC has no errors;
three DSP input-pipelining warnings and BRAM output-register methodology warnings
remain as optimization suggestions. The decimal point is intentionally tied off.

Verification already run:

- Two component suites at 8 and **868 clocks/bit**: all 256 byte values, each TX
  bit duration, RX loopback, busy writes, false starts, framing errors, reset abort,
  signed MAC/bias/enable hold, and 44 seven-segment combinations.
- Core: all **10 supplied golden images plus 4 edge cases**; exact agreement for
  **2,828 layer outputs**, **2,828 INT32 accumulators**, **10,976 normalized pixels**,
  all decisions/margins, and latency. No arithmetic mismatches.
- Requantization: **266,242 boundary-region values**, plus signed INT32 extremes,
  checking ReLU, both shift/rounding rules, saturation, and the wide rounding add.
- Full UART/top integration: **11 complete images, 110 exact returned logits**,
  ACK/response CRCs, busy request rejection without active-image corruption,
  bad CRC, incomplete timeout, wrong length, framing error, display code/hold.
- Five Python tests: USB/Bluetooth duplicate-name enumeration, standard CRC
  vector/framing, fragmented/stale responses,
  error/timeout handling with no retransmission, and a GUI FPGA-mode test that
  forbids local model use and asserts PyTorch/inference modules are not imported.

Run the same simulation and transport tests:

```powershell
.\.venv\Scripts\python.exe -m fpga_ai.scripts.run_sim
.\.venv\Scripts\python.exe -m digit_recognition.test_serial_fpga
```

The simulator runner uses the existing local Icarus installation, or accepts
`--iverilog-dir C:\path\to\iverilog\bin`. RTL compiles with `-g2001`.
Test data preparation runs the integer reference **offline for verification only**.
It verifies the original golden logits before generating edge cases. The normal
GUI does not import or run this preparation/reference code.

For real board testing after programming, with the app idle/closed:

```powershell
.\.venv\Scripts\python.exe -m fpga_ai.scripts.check_hardware --port COM6
```

Replace the port as needed. This sends 14 fixed images and compares all **140
returned logits** with prepared expected values. This client does not run a model.
It writes `physical_hardware.json` only after every check passes. Hardware Manager
programming success alone is never recorded as a successful inference test.

## UART troubleshooting

- **Device not found:** check `serial.tools.list_ports -v` and Device Manager's
  **USB Serial Port**, not a Bluetooth entry. On this laptop, both were initially
  assigned COM6, and FTDI's interface was disabled despite a Started device status.
  Assign distinct port numbers in Device Manager and reconnect/restart the USB
  device. Port Settings → Advanced may require administrator access.
  After fixing duplicate names, the FTDI child device needed an administrator
  restart before its interface became Enabled. This was done successfully for
  the connected board; no driver replacement was needed.
- **Access denied/busy:** close serial terminals or another app instance using
  the USB port. Only one outstanding request/client should use it.
- **Timeout:** confirm AI bitstream is programmed, SW0 is low, board is powered,
  and both endpoints use the same baud. A wrong/counter bitstream cannot answer
  this protocol. Keep the PROG cable connected through the entire test.
- **Unknown:** inspect the 28x28 preview and score gap. A real digit can be rejected
  and a non-digit can pass. No model can guarantee unknown detection from confidence
  alone. Hardware uses a conservative integer bound requiring >=90% softmax.

See [PROTOCOL.md](PROTOCOL.md) for byte-level framing and status codes. Reports are
in `fpga_ai/build/` and `digit_recognition/reports/`; source/build/program scripts
remain separate from generated Vivado artifacts and original counter sources.
