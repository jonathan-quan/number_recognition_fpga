"""FPGA transport only: no model, PyTorch, ONNX or local inference imports."""

from dataclasses import asdict, dataclass
import struct
import time

import numpy as np
import serial
from serial.tools import list_ports

SYNC = b"\xa5\x5a"
STATUS = {0: "result", 1: "accepted", 2: "CRC error", 3: "invalid length",
          4: "incomplete packet timeout", 5: "FPGA busy", 6: "unsupported version/type",
          7: "UART framing error"}


def crc16(data):
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def image_packet(pixels, sequence):
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8 or pixels.shape != (28, 28):
        raise ValueError("FPGA expects a uint8 [28,28] image")
    body = struct.pack("<BBBH", 1, 1, sequence, 784) + pixels.tobytes(order="C")
    return SYNC + body + struct.pack("<H", crc16(body))


@dataclass
class FPGAResult:
    digit: int
    unknown: bool
    cycles: int
    margin: int
    logits: tuple
    sequence: int

    @property
    def display(self):
        return "9999" if self.unknown else f"{self.digit:04d}"

    def as_dict(self):
        return {**asdict(self), "display": self.display, "backend": "fpga"}


def available_ports():
    # Windows can report a stale Bluetooth port with the same COM name. Prefer
    # the FTDI USB entry, but never silently choose a different physical port.
    ports = sorted(list_ports.comports(), key=lambda p: ("USB" not in p.hwid, p.device))
    unique = {}
    for port in ports:
        unique.setdefault(port.device, port)
    return list(unique.values())


class FPGAClient:
    def __init__(self, port, baud=115200, timeout=3.0):
        if not port:
            raise ValueError("Select the Basys 3 USB serial COM port")
        self.port = port
        self.baud = int(baud)
        self.timeout = timeout
        self.sequence = 0

    @staticmethod
    def _read_exact(connection, count, deadline):
        value = bytearray()
        while len(value) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for FPGA. Check bitstream, COM port, baud, power and SW0 reset.")
            value.extend(connection.read(count - len(value)))
        return bytes(value)

    def _response(self, connection, deadline):
        previous = 0
        while True:
            byte = self._read_exact(connection, 1, deadline)[0]
            if previous == 0xA5 and byte == 0x5A:
                break
            previous = byte
        header = self._read_exact(connection, 5, deadline)
        version, kind, sequence, length = struct.unpack("<BBBH", header)
        if version != 1 or kind != 0x81 or length != 52:
            raise ValueError("Invalid FPGA response header; verify matching Stage 2 firmware")
        payload = self._read_exact(connection, length, deadline)
        received_crc = struct.unpack("<H", self._read_exact(connection, 2, deadline))[0]
        if received_crc != crc16(header + payload):
            raise ValueError("FPGA response CRC mismatch")
        return sequence, payload

    def predict(self, pixels):
        sequence = self.sequence
        self.sequence = (self.sequence + 1) & 255
        packet = image_packet(pixels, sequence)
        # No automatic retransmission: avoid duplicating a possibly accepted
        # transaction after a lost ACK. The user can explicitly Predict again.
        with serial.Serial(self.port, self.baud, timeout=0.05, write_timeout=1,
                           bytesize=8, parity="N", stopbits=1, xonxoff=False, rtscts=False, dsrdtr=False) as connection:
            connection.reset_input_buffer()
            if connection.write(packet) != len(packet):
                raise OSError("Incomplete UART write")
            connection.flush()
            deadline = time.monotonic() + self.timeout
            acknowledged = False
            while True:
                response_sequence, payload = self._response(connection, deadline)
                if response_sequence != sequence:
                    continue  # stale or unrelated response
                status, digit, flags, reserved, cycles, margin, *logits = struct.unpack("<BBBBII10i", payload)
                if status == 1:
                    acknowledged = True
                    continue
                if status != 0:
                    raise RuntimeError(STATUS.get(status, f"FPGA error {status}"))
                if not acknowledged:
                    raise ValueError("FPGA result arrived without a matching acknowledgment")
                if digit > 9 or flags & ~1 or reserved:
                    raise ValueError("Invalid FPGA result fields")
                return FPGAResult(digit, bool(flags & 1), cycles, margin, tuple(logits), sequence)
