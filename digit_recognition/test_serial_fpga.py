"""Transport and FPGA-only GUI tests; no model inference is used."""
import struct
import sys
import time
import tkinter as tk
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from . import app
from .serial_fpga import FPGAClient, FPGAResult, available_ports, crc16, image_packet


def response(status=0, sequence=0):
    body=struct.pack('<BBBH',1,0x81,sequence,52)
    body+=struct.pack('<BBBBII10i',status,7 if status==0 else 255,0,0,328190,9000,*range(10))
    return b'\xa5\x5a'+body+struct.pack('<H',crc16(body))


class FakeSerial:
    def __init__(self,received):self.received=bytearray(received);self.written=b''
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def reset_input_buffer(self):pass
    def write(self,data):self.written+=data;return len(data)
    def flush(self):pass
    def read(self,count):
        count=min(count,3)  # Force fragmented reads, including split fields.
        value=bytes(self.received[:count]);del self.received[:count];return value


class TransportTests(unittest.TestCase):
    def test_duplicate_com_names_prefer_usb_description(self):
        bluetooth=SimpleNamespace(device='COM6',hwid='BTHENUM',description='Bluetooth')
        usb=SimpleNamespace(device='COM6',hwid='USB VID:PID=0403:6010',description='USB Serial Port')
        with patch('digit_recognition.serial_fpga.list_ports.comports',return_value=[bluetooth,usb]):
            ports=available_ports()
        self.assertEqual(ports,[usb])

    def test_crc_and_request_layout(self):
        self.assertEqual(crc16(b'123456789'),0x29B1)
        pixels=np.arange(784,dtype=np.uint16).astype(np.uint8).reshape(28,28)
        packet=image_packet(pixels,37)
        self.assertEqual(len(packet),793)
        self.assertEqual(packet[:7],b'\xa5\x5a\x01\x01\x25\x10\x03')
        self.assertEqual(packet[7:-2],pixels.tobytes())
        self.assertEqual(struct.unpack('<H',packet[-2:])[0],crc16(packet[2:-2]))

    def test_fragmented_ack_result_and_stale_sequence(self):
        connection=FakeSerial(b'noise'+response(1,99)+response(1)+response())
        with patch('digit_recognition.serial_fpga.serial.Serial',return_value=connection):
            result=FPGAClient('TEST').predict(np.zeros((28,28),np.uint8))
        self.assertEqual(result.display,'0007');self.assertEqual(result.logits,tuple(range(10)))
        self.assertEqual(len(connection.written),793)

    def test_errors_and_no_retransmission(self):
        for data,error in [(response(2),RuntimeError),(response(),ValueError),
                           (response(1)+response()[:-1]+b'\x00',ValueError),(b'',TimeoutError)]:
            connection=FakeSerial(data)
            with patch('digit_recognition.serial_fpga.serial.Serial',return_value=connection):
                with self.assertRaises(error):
                    FPGAClient('TEST',timeout=0.01).predict(np.zeros((28,28),np.uint8))
            self.assertEqual(len(connection.written),793)

    def test_fpga_gui_does_not_load_or_call_a_model(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('digit_recognition.model',sys.modules)
        self.assertNotIn('digit_recognition.predict',sys.modules)
        class ForbiddenModel:
            def __call__(self,*args):raise AssertionError('Local inference is forbidden')
        root=tk.Tk();root.withdraw()
        try:
            gui=app.DigitApp(root,model=ForbiddenModel(),mode='fpga',port='TEST')
            with patch.object(FPGAClient,'predict',return_value=FPGAResult(7,False,328190,9000,tuple(range(10)),0)):
                gui.predict()
                deadline=time.monotonic()+1
                while gui.pending and time.monotonic()<deadline:
                    root.update();time.sleep(0.005)
            self.assertFalse(gui.pending);self.assertEqual(gui.display_text.get(),'0007')
            self.assertNotIn('torch',sys.modules)
            self.assertNotIn('digit_recognition.integer_reference',sys.modules)
            gui.clear();self.assertEqual(gui.display_text.get(),'----')
        finally:root.destroy()


if __name__=='__main__':unittest.main(verbosity=2)
