"""Offline reference/vector preparation. Never imported by the FPGA drawing app."""
import json
from pathlib import Path
import numpy as np
from digit_recognition.integer_reference import IntegerMLP, EXPORT_DIR, read_mem

ROOT = Path(__file__).resolve().parents[1]

def write_words(path, values, bits):
    path.write_text("".join(f"{int(v)&((1<<bits)-1):0{bits//4}x}\n" for v in np.asarray(values).reshape(-1)), encoding="ascii")

def main():
    model = IntegerMLP()
    layers = model.manifest["layers"]
    assert [(v["inputs"],v["outputs"],v["right_shift"]) for v in layers] == [(784,128,10),(128,64,8),(64,10,0)]
    assert model.manifest["rejection"]["method"] == "integer_softmax_lower_bound"
    cases = [read_mem(EXPORT_DIR / f"test_vectors/sample{i}_pixels.mem",8,(28,28),signed=False).astype(np.uint8) for i in range(10)]
    cases += [np.zeros((28,28),np.uint8), np.full((28,28),255,np.uint8),
              np.random.default_rng(42).integers(0,256,(28,28),dtype=np.uint8),
              np.tile(np.arange(28,dtype=np.uint8)*9,(28,1))]
    # Saved mouse-drawn nine: was incorrectly rejected by the original gap rule.
    cases.append(read_mem(ROOT/'sim'/'regression_nine.mem',8,(28,28),signed=False).astype(np.uint8))
    traces=[]; accumulators=[]; digits=[]; unknowns=[]; margins=[]; normalized=[]
    for i,pixels in enumerate(cases):
        scores,trace=model.infer(pixels,trace=True)
        traces.extend(np.concatenate([trace["hidden1"],trace["hidden2"],scores]))
        accumulators.extend(np.concatenate([trace["acc1"],trace["acc2"],trace["acc3"]]))
        normalized.extend(trace["input"])
        digit=int(scores.argmax()); display=model.display(scores,blank=not pixels.any())
        digits.append(digit);unknowns.append(int(display=="9999"))
        order=np.sort(scores);margins.append(int(order[-1]-order[-2]))
        if i<10:
            np.testing.assert_array_equal(scores,read_mem(EXPORT_DIR / f"test_vectors/sample{i}_logits.mem",32,(10,)))
    out=ROOT/'sim'/'data';out.mkdir(parents=True,exist_ok=True)
    for name,value,bits in [('pixels',cases,8),('normalized',normalized,8),('trace',traces,32),('accumulators',accumulators,32),('digits',digits,8),('unknowns',unknowns,8),('margins',margins,32)]:
        write_words(out/f'{name}.mem',value,bits)
    confidence_cases=[[-9012,-15034,-6164,7282,-10053,-320,-20202,-7816,4317,9910],
                      [0]*10,[0]*9+[2628],[-2147483648]*9+[2147483647]]
    for gap in [0,1,15,16,2239,2240,2249,2250,2255,2256,2260,4500,16383,16384,16385,100000]:
        confidence_cases.append([0]+[-100000]*8+[gap])
    confidence_cases+=np.random.default_rng(8).integers(-30000,30000,(20,10)).tolist()
    write_words(out/'confidence_scores.mem',confidence_cases,32)
    write_words(out/'confidence_sums.mem',[model.exponential_sum(row) for row in confidence_cases],32)
    write_words(out/'confidence_unknown.mem',[int(model.display(row)=='9999') for row in confidence_cases],8)
    (out/'cases.json').write_text(json.dumps({'cases':len(cases),'golden':10,'trace_values_per_image':202,'expected_inference_cycles':328190},indent=2))
    print(f'Prepared {len(cases)} cases; original 10 golden logits verified against integer reference.')

if __name__=='__main__':main()
