"""Power-law (gamma) transformation: s = c r^gamma."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import apply_lut, result


def run(img, params):
    gamma = max(float(params.get("gamma", 0.6)), 1e-6)
    c = float(params.get("c", 1.0))
    r = np.arange(256, dtype=np.float64) / 255.0
    lut = np.clip(np.rint(255.0 * c * np.power(r, gamma)),
                  0, 255).astype(np.uint8)
    out = apply_lut(img, lut)
    if gamma < 1:
        effect = "γ < 1 brightens the image (expands dark values)."
    elif gamma > 1:
        effect = "γ > 1 darkens the image (compresses dark values)."
    else:
        effect = "γ = 1 leaves the image unchanged."
    log = (
        f"s = c · r^γ,   γ = {gamma:g}, c = {c:g}\n"
        f"{effect}\n"
        "Normalise r to 0..1 first, apply the power, then scale back to\n"
        "0..255.  Gamma is the standard monitor / camera response control."
    )
    return result(out, log)


OPERATION = Operation(
    key="gamma",
    name="Power-law (gamma)",
    category="Point Processing",
    description=(
        "s = c · r^γ after normalising r to 0..1. γ < 1 brightens, "
        "γ > 1 darkens."
    ),
    func=run,
    params=(
        Param("gamma", "γ", "float", 0.6, 0.05, 5.0, 0.05, 2),
        Param("c", "c", "float", 1.0, 0, 3, 0.05, 2),
    ),
    pointwise=True,
)
