"""Contrast stretching with optional percentile clipping."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import as_float, as_uint8, result


def run(img, params):
    low = float(params.get("low", 0.0))
    high = float(params.get("high", 100.0))
    a = as_float(img)
    lo = float(np.percentile(a, low))
    hi = float(np.percentile(a, high))
    if hi <= lo:
        log = (
            f"Percentiles {low:g}%..{high:g}% give {lo:.4g}..{hi:.4g}.\n"
            "The range is empty, so the image was left unchanged."
        )
        return result(a, log)
    out = as_uint8((a - lo) / (hi - lo) * 255.0)
    log = (
        "X_new = (X - X_min) / (X_max - X_min) · 255\n"
        f"Using percentiles {low:g}% and {high:g}%:\n"
        f"  X_min = {lo:.4g},  X_max = {hi:.4g}\n"
        "A pure min-max stretch uses 0% and 100%.  Moving the percentiles\n"
        "inward clips a few outliers so they cannot ruin the range."
    )
    return result(out, log)


OPERATION = Operation(
    key="contrast",
    name="Contrast stretching",
    category="Point Processing",
    description=(
        "X_new = (X - Xmin)/(Xmax - Xmin) · 255. Scales the histogram "
        "wider; percentile clipping avoids outliers ruining the range."
    ),
    func=run,
    params=(
        Param("low", "Low percentile", "float", 0, 0, 49, 0.5, 1),
        Param("high", "High percentile", "float", 100, 51, 100, 0.5, 1),
    ),
)
