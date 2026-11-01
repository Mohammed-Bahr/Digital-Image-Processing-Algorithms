"""Negative image: s = (L - 1) - r."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation
from ..core.utils import apply_lut, result


def run(img, params):
    lut = np.arange(255, -1, -1, dtype=np.uint8)
    out = apply_lut(img, lut)
    log = (
        "s = (L - 1) - r = 255 - r\n"
        "Dark becomes light and light becomes dark.  Useful for detail\n"
        "that is white or gray on a dark background (mammograms, etc.).\n"
        "The transfer curve is a straight line with slope -1."
    )
    return result(out, log)


OPERATION = Operation(
    key="negative",
    name="Negative",
    category="Point Processing",
    description=(
        "s = (L - 1) - r. Reverses brightness; useful when detail is "
        "white/gray on a dark background."
    ),
    func=run,
    pointwise=True,
)
