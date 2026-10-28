"""Offline rejection evaluation; never trains or exports model weights."""
import json

import numpy as np

from .download_model import ROOT
from .integer_reference import IntegerMLP
from .mnist_data import test_set


def main():
    model = IntegerMLP()
    pixels, labels = test_set()
    logits = np.concatenate([model.infer(pixels[i:i+256]) for i in range(0,len(pixels),256)])
    ordered = np.sort(logits,axis=1)
    old = ordered[:,-1] - ordered[:,-2] >= 4500
    new = np.array([model.display(row) != '9999' for row in logits])
    correct = logits.argmax(axis=1) == labels
    exp = np.exp((logits-logits.max(axis=1,keepdims=True))/1024)
    confidence = 1/exp.sum(axis=1)
    assert np.all(confidence[new] >= 0.90)
    report = {'samples':len(pixels),'weights_and_neural_arithmetic':'unchanged','old':{},'new':{}}
    for name, accepted in [('old',old),('new',new)]:
        nines = labels == 9
        report[name] = {'accepted':int(accepted.sum()),
                        'accepted_accuracy':float(correct[accepted].mean()),
                        'nines_rejected':int((nines & ~accepted).sum()),
                        'nines_total':int(nines.sum())}
    report['new']['minimum_exact_softmax_among_accepted'] = float(confidence[new].min())
    report['note'] = 'MNIST digit-only evaluation; does not measure non-digit performance. No training or model export.'
    target = ROOT / 'reports' / 'rejection_confidence_evaluation.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
