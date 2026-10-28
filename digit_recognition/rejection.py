"""Offline LUT construction and integer-only confidence rejection constants.

The normal FPGA drawing app does not import this module or compute confidence.
The FPGA sums conservative upper bounds on exp(score[i]-best), scaled by 65536.
The winner's probability is >= 0.90 when that sum is <= floor(65536/0.90).
"""

from decimal import Decimal, ROUND_CEILING, localcontext

SCALE = 65536
LUT_DEPTH = 1024
DELTA_SHIFT = 4  # Q10 logits: 16 units = 1/64 real logit.
MAXIMUM_SUM = 72817
LUT_FILE = "confidence_exp.mem"


def make_exp_lut():
    """Offline only: ceil(65536 * exp(-index/64)) with high precision."""
    with localcontext() as context:
        context.prec = 50
        return [int((Decimal(SCALE) * (-Decimal(i) / 64).exp()).to_integral_value(rounding=ROUND_CEILING))
                for i in range(LUT_DEPTH)]


def configuration():
    return {
        "method": "integer_softmax_lower_bound",
        "minimum_confidence_numerator": 9,
        "minimum_confidence_denominator": 10,
        "exp_lut_file": LUT_FILE,
        "exp_lut_bits": 17,
        "exp_lut_depth": LUT_DEPTH,
        "exp_scale": SCALE,
        "delta_right_shift": DELTA_SHIFT,
        "tail_upper_bound": 1,
        "maximum_exponential_sum": MAXIMUM_SUM,
        "note": "Floor delta to 1/64-logit bins and ceil LUT values; acceptance conservatively implies softmax >=0.90. Blank always rejects.",
        "previous_gap_only_threshold": 4500,
    }
