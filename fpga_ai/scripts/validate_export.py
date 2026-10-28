"""Fail early if the exported model no longer matches the RTL arithmetic contract."""
import hashlib
import json
from pathlib import Path

def main():
    folder=Path(__file__).resolve().parents[2]/'digit_recognition'/'export'
    manifest=json.loads((folder/'manifest.json').read_text())
    expected=[(784,128,10),(128,64,8),(64,10,0)]
    assert [(v['inputs'],v['outputs'],v['right_shift']) for v in manifest['layers']]==expected,'Unsupported layer dimensions/shifts'
    assert manifest['weight_bits']==8 and manifest['bias_bits']==32 and manifest['accumulator_bits']==32
    assert manifest['hidden_saturation']==[0,127]
    assert manifest['rounding']=='nearest, exact halves away from zero'
    assert manifest['rejection']['method']=='integer_softmax_lower_bound'
    assert manifest['rejection']['maximum_exponential_sum']==72817
    assert manifest['rejection']['delta_right_shift']==4 and manifest['rejection']['exp_scale']==65536
    assert manifest['layers'][-1]['output_fraction_bits']==10,'Confidence LUT assumes Q10 logits'
    assert manifest['rejection']['exp_lut_bits']==17 and manifest['rejection']['exp_lut_depth']==1024
    assert manifest['rejection']['tail_upper_bound']==1
    lengths={'input_lut.mem':256,'fc1_weights.mem':100352,'fc2_weights.mem':8192,
             'fc3_weights.mem':640,'fc1_biases.mem':128,'fc2_biases.mem':64,'fc3_biases.mem':10,
             'confidence_exp.mem':1024}
    for name,count in lengths.items():
        raw=(folder/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest['mem_sha256'][name],f'Checksum mismatch: {name}'
        assert len(raw.split())==count,f'Word-count mismatch: {name}'
    print('EXPORT_CONTRACT_PASS: dimensions, scaling shifts, rounding, confidence rejection, 8 memory hashes/lengths')

if __name__=='__main__':main()
