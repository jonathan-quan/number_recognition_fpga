"""Verify the FPGA confidence rule without training or changing model tensors."""
import json
import unittest
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .integer_reference import IntegerMLP
from .rejection import make_exp_lut


class ConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = IntegerMLP()

    def test_saved_nine_is_accepted_with_identical_logits(self):
        logits = [-9012,-15034,-6164,7282,-10053,-320,-20202,-7816,4317,9910]
        self.assertEqual(self.net.exponential_sum(logits),70881)
        self.assertEqual(self.net.display(logits),'0009')
        self.assertEqual(self.net.display(logits,blank=True),'9999')

    def test_large_top_two_gap_alone_is_not_sufficient(self):
        # Nine close competitors: a 2.57-logit gap still only gives ~59%.
        logits = [0]*9+[2628]
        self.assertEqual(self.net.display(logits),'9999')
        self.assertEqual(self.net.display([0]*10),'9999')

    def test_lut_upper_bounds_and_extremes(self):
        expected=make_exp_lut()
        np.testing.assert_array_equal(self.net.exp_lut,expected)
        self.assertEqual(expected[0],65536)
        self.assertTrue(all(a>=b>=1 for a,b in zip(expected,expected[1:])))
        self.assertEqual(self.net.display([-2147483648]*9+[2147483647]),'0009')
        # A quantized bin and the clipped tail must both overestimate exp(-delta).
        with localcontext() as context:
            context.prec=40
            for delta in [0,1,15,16,2240,2250,2255,2256,16383,16384,100000]:
                term=expected[delta>>4] if delta<16384 else 1
                self.assertGreaterEqual(Decimal(term),Decimal(65536)*(-Decimal(delta)/1024).exp())

    def test_acceptance_implies_at_least_ninety_percent(self):
        random=np.random.default_rng(91)
        for _ in range(2000):
            logits=random.integers(-20000,20000,10,dtype=np.int64)
            exp=np.exp((logits-logits.max())/1024)
            exact_confidence=1/exp.sum()
            if self.net.display(logits)!='9999':
                self.assertGreaterEqual(exact_confidence,0.9)


if __name__=='__main__':unittest.main(verbosity=2)
