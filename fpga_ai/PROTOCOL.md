# UART protocol version 1

100 MHz FPGA clock, default **115200 baud, 8 data bits, no parity, 1 stop bit**.
No RTS/CTS, XON/XOFF, or DTR-based reset. Change both the bitstream's `BAUD`
parameter and Python baud setting together if using a different rate.

## Request

Exactly 793 bytes per image:

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 2 | Sync `A5 5A` |
| 2 | 1 | Version `01` |
| 3 | 1 | Command `01`: classify image |
| 4 | 1 | Sequence ID, 0..255 |
| 5 | 2 | Little-endian payload length, `10 03` = 784 |
| 7 | 784 | Raw uint8 grayscale, white ink on black, row-major `y*28+x` |
| 791 | 2 | CRC16, low byte first |

CRC16/CCITT-FALSE: polynomial 0x1021, initial value 0xFFFF, no reflection, XOR-out
0x0000. Cover bytes 2 through 790, including version, type, sequence, length and
image. Exclude sync and the CRC itself. Check vector `123456789` -> `29B1`.
Do not send normalized signed bytes; normalization happens through the FPGA LUT.

## Response

Exactly 61 bytes: same 7-byte header, version `01`, response type `81`, echoed
sequence, payload length 52 (`34 00`), payload, and CRC over header bytes 2 onward
plus payload. All multibyte fields use little-endian order.

| Payload offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 1 | Status |
| 1 | 1 | Top digit 0..9 for result; `FF` for ACK/error |
| 2 | 1 | Result flags: bit 0 = unknown, other bits zero |
| 3 | 1 | Reserved, zero |
| 4 | 4 | Unsigned inference clock count |
| 8 | 4 | Unsigned top-two integer logit gap |
| 12 | 40 | Ten signed INT32 logits, class order 0..9 |

Non-result responses zero the cycle, margin, logit and flag fields. An accepted
image receives **ACK then RESULT**, with matching sequence IDs and valid CRCs.
Digit and unknown flag are calculated by hardware. Python uses these directly;
the normal app does not calculate softmax, argmax, or run a local model.

| Status | Meaning |
|---:|---|
| 0 | RESULT |
| 1 | ACK: complete valid image accepted, inference started |
| 2 | CRC error: image not used for inference |
| 3 | Invalid payload length |
| 4 | Incomplete packet: interbyte timeout (100 ms default) |
| 5 | Busy: wait for outstanding result, then explicitly retry |
| 6 | Unsupported version or command |
| 7 | UART framing error |

## Transaction and recovery rules

Only one inference transaction is outstanding. The FPGA reserves the image buffer
through inference and result transmission. A request arriving while occupied is
drained and answered BUSY, without changing the active buffer. Four small FIFO
entries hold ACK/error responses; an independent pending flag retains the result
while ACK is transmitted. Do not flood malformed headers faster than responses
can be transmitted; an exhausted response FIFO may drop an error response, and
the host must time out and resynchronize.

For an idle request, incoming pixels write a staging image buffer while CRC is
calculated. There is no `start` pulse until all 784 bytes and the CRC are valid.
Invalid partial contents cannot trigger inference or change the held display.
The next complete request overwrites all 784 addresses. A detected UART framing
error or interbyte timeout abandons the partial frame and resumes sync search.
Invalid length/version headers also resume sync search immediately.

Python waits for a matching ACK and RESULT, checks framing/length/CRC, and reports
timeout or FPGA errors without local inference fallback. It does not automatically
retransmit a request after an ambiguous lost response. Press Predict again to retry.

FPGA reset abandons all transactions and displays `9999`. The host can retry after
reset is released. The display otherwise holds its completed result across packet
errors, UART idle time, and subsequent inference until the next result completes.

Blank preprocessing returns an all-zero pixel image. Hardware independently
tracks whether any raw pixel is nonzero; an all-zero image is always unknown.
For nonblank images, the FPGA now evaluates a conservative integer softmax
confidence bound and requires at least 90%. It sums a 17-bit exponential LUT over
all ten scores, with 20-bit accumulation. A sum above 72,817 rejects the input.
The reported top-two margin remains diagnostic; it no longer determines rejection.
Packet framing and response fields are unchanged. This replaces the original
gap-only threshold of 4,500, which rejected some valid high-confidence digits.
Unknown detection remains imperfect for letters and scribbles.
