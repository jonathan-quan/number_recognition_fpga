"""Send fixed test images to the real FPGA and compare returned golden logits.

This verification client does NOT compute a neural network on the laptop.
Expected integers were generated separately from the existing reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from digit_recognition.serial_fpga import FPGAClient

ROOT=Path(__file__).resolve().parents[1]

def read_words(path,bits):
    values=np.array([int(v,16) for v in path.read_text().split()],dtype=np.int64)
    if bits==32:values=np.where(values>=2**31,values-2**32,values)
    return values

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--baud',type=int,default=115200)
    args=parser.parse_args()
    data=ROOT/'sim'/'data'
    pixels=read_words(data/'pixels.mem',8).astype(np.uint8).reshape(-1,28,28)
    expected=read_words(data/'trace.mem',32).reshape(-1,202)[:,192:]
    digits=read_words(data/'digits.mem',8)
    unknowns=read_words(data/'unknowns.mem',8)
    margins=read_words(data/'margins.mem',32)
    client=FPGAClient(args.port,args.baud)
    measurements=[]
    for i,image in enumerate(pixels):
        started=time.monotonic()
        result=client.predict(image)
        elapsed=(time.monotonic()-started)*1000
        np.testing.assert_array_equal(result.logits,expected[i],err_msg=f'Physical FPGA sample {i}')
        assert result.digit==int(digits[i]) and result.unknown==bool(unknowns[i]),f'Decision mismatch sample{i}'
        assert result.margin==int(margins[i]),f'Margin mismatch sample{i}'
        assert result.cycles==328190,f'Latency mismatch sample{i}'
        measurements.append({'sample':i,**result.as_dict(),'round_trip_ms':round(elapsed,3),'exact_logits':True})
        print(f'HARDWARE_PASS sample={i} display={result.display} clocks={result.cycles} round_trip_ms={elapsed:.1f}',flush=True)
    bitfile=ROOT/'build'/'digit_ai_top.bit'
    report={'verification':'physical Basys 3 USB-UART, not simulation','port':args.port,'baud':args.baud,
            'images_passed':len(pixels),'exact_logits':len(pixels)*10,'cycles_per_inference':328190,
            'bitstream_sha256':hashlib.sha256(bitfile.read_bytes()).hexdigest(),
            'cases':measurements}
    target=ROOT.parent/'digit_recognition'/'reports'/'physical_hardware.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'PASS: {len(pixels)} physical FPGA images, {len(pixels)*10} exact integer logits. Report: {target}',flush=True)

if __name__=='__main__':main()
