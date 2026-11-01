"""First-derivative edge magnitude (Sobel or Scharr)."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import as_uint8, describe_kernel, result
from .common import (gradients, orientation_rgb, SOBEL_X, SOBEL_Y,
                     SCHARR_X, SCHARR_Y)


def run(img, params):
    variant = str(params.get("variant", "Sobel"))
    threshold = float(params.get("threshold", 25.0))
    gx, gy, mag, angle = gradients(img, variant)
    peak = float(mag.max())
    if peak <= 0:
        out = np.zeros(mag.shape, dtype=np.uint8)
    else:
        m8 = mag / peak * 255.0
        if threshold > 0:
            m8 = np.where(m8 >= threshold, m8, 0.0)
        out = as_uint8(m8)

    kx, ky = (SCHARR_X, SCHARR_Y) if variant.lower().startswith("scharr") \
        else (SOBEL_X, SOBEL_Y)
    log = (
        f"{variant} first derivatives on the luminance plane:\n"
        f"Gx = {variant}_x ⊛ f,   Gy = {variant}_y ⊛ f\n"
        "G = √(Gx² + Gy²)\n"
        "Flat regions have Gx = Gy = 0; edges give a large response.\n"
        "The result is scaled to 0..255 for display"
        + (f" and thresholded at {threshold:g}." if threshold > 0 else ".")
        + "\n\nGx mask:\n" + describe_kernel(kx)
        + "\n\nGy mask:\n" + describe_kernel(ky)
    )
    return result(out, log, kernel=kx,
                  aux={"Gradient direction": orientation_rgb(angle, mag)})


OPERATION = Operation(
    key="sobel",
    name="Sobel magnitude",
    category="Sharpening",
    description=(
        "First derivative: G = √(Gx² + Gy²). Bright edges on black; "
        "flat regions respond with 0."
    ),
    func=run,
    params=(
        Param("variant", "Operator", "choice", "Sobel",
              choices=("Sobel", "Scharr")),
        Param("threshold", "Threshold", "float", 25, 0, 255, 1, 0),
    ),
)
