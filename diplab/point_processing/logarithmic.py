"""Logarithmic point operation: s = c log(1 + r)."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import apply_lut, as_float, result


def run(img, params):
    c = float(params.get("c", 0.0))
    peak = float(as_float(img).max())
    if c <= 0:
        c = 255.0 / np.log1p(max(peak, 1.0))
        auto = " (auto: c = 255 / log(1 + r_max))"
    else:
        auto = ""
    lut = np.clip(np.rint(c * np.log1p(np.arange(256, dtype=np.float64))),
                  0, 255).astype(np.uint8)
    out = apply_lut(img, lut)
    log = (
        f"s = c · log(1 + r),   c = {c:.4g}{auto}\n"
        "The curve rises steeply at low r and flattens at high r, so dark\n"
        "detail is expanded and bright detail is compressed.  The same\n"
        "idea is used to display Fourier spectra (log(1 + |F|))."
    )
    return result(out, log)


OPERATION = Operation(
    key="log",
    name="Logarithmic",
    category="Point Processing",
    description=(
        "s = c · log(1 + r). Expands dark values and compresses bright "
        "ones; also used to display Fourier spectra."
    ),
    func=run,
    params=(Param("c", "c (0 = auto)", "float", 0.0, 0, 10, 0.05, 3),),
    pointwise=True,
)
