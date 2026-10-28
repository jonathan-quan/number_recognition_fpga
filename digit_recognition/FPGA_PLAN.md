# Basys 3 integer arithmetic specification and original deployment plan

Stage 2 now implements this arithmetic in `fpga_ai/rtl/digit_ai_top.v` and its
supporting modules. See [the Stage 2 guide](../fpga_ai/README.md) for actual
simulation, implementation/timing results, deployment, and hardware-test status.
The neural-network numerical contract below remains valid. The original gap-only
rejection was replaced with the confidence check described below. The resource estimates and "Later
integration sequence" are the original planning record, superseded by measured
results in the Stage 2 guide. The original counter project remains untouched.

## Exported arithmetic contract

`export/manifest.json` is the machine-readable source of truth. Values below
describe the checked-in export, calibrated on 2,048 unlabeled MNIST images.
Every tensor uses `real_value ~= stored_integer / 2^fraction_bits`.

| Tensor | Storage | Fraction bits | Real step |
|---|---|---:|---:|
| Normalized input | signed INT8 | 5 | 1/32 |
| FC1 weights | signed INT8 | 7 | 1/128 |
| FC1 biases/accumulator | signed INT32 | 12 | 1/4096 |
| Hidden 1 after ReLU | signed INT8, 0..127 | 2 | 1/4 |
| FC2 weights | signed INT8 | 8 | 1/256 |
| FC2 biases/accumulator | signed INT32 | 10 | 1/1024 |
| Hidden 2 after ReLU | signed INT8, 0..127 | 2 | 1/4 |
| FC3 weights | signed INT8 | 8 | 1/256 |
| FC3 biases/final logits | signed INT32 | 10 | 1/1024 |

A fraction-bit count of 8 on an 8-bit weight means a 1/256 step and range
-0.5..0.49609375; it does not add a ninth storage bit.

For each output neuron, initialize the signed accumulator to its bias, then add
`signed_activation * signed_weight` over all input neurons. FC1 applies ReLU,
rounds right by 10 bits, and saturates to 0..127. FC2 does the same with a shift
of 8 bits. FC3 keeps signed 32-bit logits without ReLU or another requantization.
Do not narrow/truncate an accumulator after each multiply-add.

Rounding is nearest, exact halves away from zero. Since hidden layers apply ReLU
before rounding, RTL can use a nonnegative wide intermediate:

```text
if accumulator <= 0: activation = 0
else: activation = min(127, (accumulator + 2^(shift-1)) >> shift)
```

Keep the addition wide enough to avoid overflow. A bare arithmetic `>>> shift`
does not reproduce this rounding. Offline weight/bias conversion uses the same
tie rule. The Python reference uses int64 internally to detect out-of-range
results before checking the planned signed INT32 limit; all inference arithmetic
itself is integer-only. It does not import PyTorch.

`input_lut.mem` has 256 entries indexed by the received raw grayscale byte:

```text
lut[p] = round_half_away_from_zero(((p/255 - 0.1307)/0.3081) * 32)
```

This floating expression is computed offline during export. The FPGA only reads
the ROM, producing signed integers from -14 to 90. Sending raw pixels therefore
keeps the laptop's future responsibilities to drawing/preprocessing/transport;
the FPGA performs normalization lookup and the entire neural-network inference.

## Memory files and golden vectors

All words are hexadecimal, one per line, without `0x` prefixes. Negative values
are two's complement. Weight shape is **[output neuron, input neuron]**, exactly
the PyTorch storage order. Address = `output_neuron * number_of_inputs + input_neuron`.
Pixels flatten as `y*28+x`.

| File | Words | Width |
|---|---:|---:|
| `fc1_weights.mem` | 100,352 | 8 |
| `fc1_biases.mem` | 128 | 32 |
| `fc2_weights.mem` | 8,192 | 8 |
| `fc2_biases.mem` | 64 | 32 |
| `fc3_weights.mem` | 640 | 8 |
| `fc3_biases.mem` | 10 | 32 |
| `input_lut.mem` | 256 | 8 |

Later, Verilog-2001 can declare a signed 8-bit ROM and initialize it with
`$readmemh("fc1_weights.mem", weight_rom)`. Add the files to the new Vivado project
as memory initialization sources and implement a synchronous ROM read to infer
BRAM. A hex byte such as `f2` means signed -14, not unsigned 242. Explicitly declare
signed operands, widen the 16-bit product, and sign-extend it into the accumulator.
Account for the BRAM read latency in valid/control signals.

`export/test_vectors/` contains ten raw image memories, LUT outputs, hidden layer
outputs, final INT32 logits, labels, and expected display values. First simulate
each layer against these vectors, then compare end to end. The Python reference
reads the actual exported memories and verifies their hashes. It can be used
independently of the float model:

```python
import numpy as np
from digit_recognition.integer_reference import IntegerMLP, read_mem, EXPORT_DIR

net = IntegerMLP()
pixels = read_mem(EXPORT_DIR / 'test_vectors/sample0_pixels.mem',
                  8, (28, 28), signed=False).astype(np.uint8)
logits = net.infer(pixels)  # integer values only
print(int(logits.argmax()), net.display(logits))
```

## Current confidence rejection on hardware

The original 4,500-unit top-two gap test below was overly conservative: a saved
mouse-drawn 9 ranked class 9 first with about 92.5% softmax confidence, yet its
2,628-unit gap failed the old rule. The neural-network weights, biases, scales,
rounding, activations and ten logits remain unchanged. Only rejection changed.

The FPGA computes an integer softmax lower bound, using Q10 output scores:

```text
delta[i] = best_score - score[i]                    # signed 33-bit subtraction
index[i] = delta[i] >> 4                           # floor to 1/64-logit bins
term[i]  = confidence_exp[index[i]] if index < 1024 else 1
sum      = sum(term[0..9])                         # unsigned 20-bit sum
unknown  = blank or sum > 72817
```

The ROM has 1,024 unsigned 17-bit words, generated offline as
`ceil(65536 * exp(-index/64))`. The top score contributes exactly 65,536.
Flooring the input delta and rounding terms upward makes every term an upper
bound; beyond a 16-logit difference, 1 remains an upper bound. Thus
`65536/sum` is a lower bound on the true softmax confidence. Comparing the sum
to floor(65536/0.9)=72,817 guarantees accepted nonblank inputs have at least 90%
model softmax confidence. The FPGA uses no division or floating-point arithmetic.
The ROM costs one additional RAMB18 and the controller takes 21 extra clocks.
The updated integer reference reads the same ROM and produces the same decision.

This is a conservative approximation to the original GUI confidence rule, with
small additional rejection near 90%. It does not establish that the input is a
digit; confident letters/scribbles may still pass. The normal laptop app only
uses the FPGA's returned digit and unknown flag and runs no neural-network or
confidence calculation.

## Original gap-only rejection (superseded)

Argmax of the ten signed logits is the predicted digit. Softmax is unnecessary
for argmax. For the prototype rejection circuit, compare the largest and second
largest integer logits; require a gap of at least **4,500**, and reject blank input.
Use lowest-index tie breaking to match the reference.
Widen the top-two subtraction to 33 bits for a general signed INT32 interface,
or prove its narrower range from the exported bounds; do not let it wrap.

This gap is ceil(ln(81)*1024). If all nine competing logits trail the winner by
at least ln(81), its softmax score is at least 0.90. It is a **sufficient,
conservative** rule, not an exact reproduction of the GUI's confidence rule.
The test-set acceptance drops from 95.58% (integer logits with GUI softmax checks)
to 90.77% with this hardware gap rule. Neither rule proves that an input is a digit.

Blank/tiny gestures become an all-zero image in laptop preprocessing. The FPGA
can independently check that the received image contains enough grayscale ink
before the input LUT; a full-zero buffer must always yield `9999`. Keep any more
elaborate ink threshold consistent with preprocessing and test it on saved images.
Do not rely on the model's prediction for an empty image.

Accepted digit d drives four BCD nibbles `000d`; unknown drives `9999`. Nine remains
`0009`, so it is distinct from unknown. Inspect the existing display module's input
encoding, segment/anode polarities, leading-zero handling, reset, and refresh timing.
The Basys 3 has multiplexed common-anode displays; the adapter must preserve all
four leading zeros where requested.

## Resource estimates

The [Digilent Basys 3 manual](https://digilent.com/reference/_media/basys3:basys3_rm.pdf)
lists 1,800 Kibits of BRAM and 90 DSP slices. The XC7A35T has 50 RAMB36 blocks;
see [AMD UG473](https://docs.amd.com/api/khub/documents/9gZGbqBxtlKXxBfkBt~lAg/content).
That is 230,400 physical RAM bytes including parity capacity. FP32 parameter data
alone is 437,544 bytes and cannot all fit on chip. The INT8/INT32 design is a
plausible fit without switching models.

- Weights: 109,184 bytes; biases: 808 bytes; normalization LUT: 256 bytes.
- Image: 784 bytes; hidden buffers: 128 + 64 bytes (possibly duplicated for ports).
- At 4,096 x 9 configuration, each RAMB36 holds 4,096 8-bit weights, with parity
  capacity unused. Separate single-byte-wide layer ROMs need approximately
  25 + 2 + 1 = **28 RAMB36 equivalents for weights**. A packed monolithic ROM can
  save some padding, but needs addressing/port design.
- Budget roughly **30-35 of 50 RAMB36 equivalents overall**, allowing for buffers,
  small ROMs, padding and moderate banking. This is an estimate, not a synthesis
  result. Four simultaneous weight reads may require banking, wider words, or
  replication; avoid blindly replicating the full model four times.
- Start with one reused 8x8 signed MAC: approximately **one DSP48E1**; a four-PE
  design might use four. Mapping must be checked in Vivado. DSP48E1 supports a
  signed 25x18 multiply and 48-bit accumulator, so the planned widths fit within
  one slice's arithmetic resources: [AMD UG479](https://docs.amd.com/api/khub/documents/gu4oRPFEh_Pm2uaAlfY6Kg/content).
- Choose **32-bit signed accumulators** and 16-bit products. The exported actual
  weights and bounded activations give conservative absolute accumulator bounds
  of 557,413 / 438,180 / 263,180, requiring at least 21 / 20 / 20 signed bits.
  These include quantized biases. Keep 32 bits for a simple reusable interface.
- There are **109,184 MACs per image**. At an assumed 100 MHz, one MAC per cycle
  has a 1.09 ms arithmetic lower bound, plus memory/control/pipeline overhead.
  Timing is not verified. A normal 2x2 matrix-multiply array can be underused by
  a single matrix-vector inference; four PEs do not automatically give 4x speed.

## Later integration sequence

1. Obtain and inspect `processing_element.v`, the array, UART and display modules
   read-only. Confirm signed operand widths, product/accumulator widths, bias
   initialization, clear/enable/valid/done signals, latency, overflow/truncation,
   matrix layout, and zero-padding behavior. Existing compatibility is **unverified**.
2. Create a separate neural-network Vivado project for `xc7a35tcpg236-1`, new top
   and constraints. Reuse compatible sources without editing the counter project.
3. Implement one serial MAC layer first and match every golden intermediate
   value. Then consider the 2x2 array with a documented tiling/feed/drain schedule.
   Fully connected inference is `y[o] = bias[o] + sum_i W[o,i]*x[i]`.
4. Add BRAM weight ROMs and ping-pong activation buffers. Switch layer dimensions,
   biases and shifts under a small controller. Hold a completed prediction until
   the next valid frame finishes.
5. Define and simulate a framed UART protocol before changing either endpoint.
   Suggested request: sync bytes, version, sequence number, 16-bit length=784,
   784 raw row-major grayscale bytes, and CRC16; response: sequence, status,
   digit/unknown and optional integer margin. Specify byte order and CRC variant
   explicitly at implementation time. Handle framing errors, dropped bytes,
   timeout, reset, and busy/ACK so partial images cannot trigger inference.
6. Connect the laptop to the shared PROG/JTAG/UART micro-USB connector (J4), using
   its Windows COM port, not the USB host connector. Proposed 115200 baud, 8-N-1
   must match existing UART timing. The payload alone takes about 68.1 ms, so UART
   likely dominates the initial latency. Stage 1 opens no COM port.
7. Add FPGA blank/gap rejection and the BCD display adapter, then verify every
   accepted digit and `9999`. Compare returned FPGA logits/decisions with the
   integer reference on saved drawings and test vectors before claiming success.

The laptop ultimately draws, resizes/centers, and sends pixels. The FPGA performs
the normalization lookup, three MAC layers, activations, argmax/rejection and display.
No PyTorch or ONNX Runtime is placed inside the FPGA.
