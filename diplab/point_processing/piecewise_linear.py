"""Piecewise-linear point transformation through two control points."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import apply_lut, result


def run(img, params):
    r1 = float(params.get("r1", 70))
    s1 = float(params.get("s1", 30))
    r2 = float(params.get("r2", 180))
    s2 = float(params.get("s2", 225))
    if not (1 <= r1 < r2 <= 254):
        raise ValueError("control points need 1 <= r1 < r2 <= 254")

    x = np.arange(256, dtype=np.float64)
    lut = np.empty(256, dtype=np.float64)

    m0 = x <= r1
    lut[m0] = (s1 / r1) * x[m0]

    m1 = (x > r1) & (x <= r2)
    lut[m1] = ((s2 - s1) / (r2 - r1)) * (x[m1] - r1) + s1

    m2 = x > r2
    lut[m2] = ((255.0 - s2) / (255.0 - r2)) * (x[m2] - r2) + s2

    table = np.clip(np.rint(lut), 0, 255).astype(np.uint8)
    out = apply_lut(img, table)
    log = (
        "Three linear segments through the control points:\n"
        f"  (0,0) -> ({r1:g},{s1:g}) -> ({r2:g},{s2:g}) -> (255,255)\n"
        "A steep middle segment stretches the mid-range (more contrast);\n"
        "shallow end segments compress the shadows and highlights."
    )
    return result(out, log)


OPERATION = Operation(
    key="piecewise",
    name="Piecewise linear",
    category="Point Processing",
    description=(
        "Three linear segments through control points (r1,s1) and "
        "(r2,s2). A steep middle segment stretches mid-range contrast."
    ),
    func=run,
    params=(
        Param("r1", "r1", "int", 70, 1, 250, 1),
        Param("s1", "s1", "int", 30, 0, 255, 1),
        Param("r2", "r2", "int", 180, 2, 254, 1),
        Param("s2", "s2", "int", 225, 0, 255, 1),
    ),
    pointwise=True,
)
