"""Laplacian sharpening: g = f - c * Laplacian(f)."""
from __future__ import annotations

from ..core.types import Operation, Param
from ..core.utils import (as_float, as_uint8, correlate, describe_kernel,
                          result)
from .common import LAPLACIANS, laplacian_abs


def run(img, params):
    kind = str(params.get("kind", "4-neighbor"))
    strength = float(params.get("strength", 1.0))
    if kind not in LAPLACIANS:
        raise ValueError(f"unknown Laplacian mask: {kind}")
    k = LAPLACIANS[kind]
    lap = correlate(img, k)
    sign = -1.0 if k[k.shape[0] // 2, k.shape[1] // 2] < 0 else 1.0
    out = as_uint8(as_float(img) + sign * strength * lap)
    log = (
        "g = f - c · ∇²f   (negative-centre mask)\n"
        f"Mask: {kind}, c = {strength:g}\n"
        "Subtracting the Laplacian adds back the high-frequency detail\n"
        "that the second derivative exposes, so edges become crisper.\n"
        "The result is clipped to 0..255; large c can create halos.\n\n"
        + describe_kernel(k)
    )
    return result(out, log, kernel=k,
                  aux={"Laplacian": laplacian_abs(img, kind)})


OPERATION = Operation(
    key="laplacian_sharp",
    name="Laplacian sharpening",
    category="Sharpening",
    description=(
        "g = f - c·∇²f for a negative-centre mask, then clip to 0..255."
    ),
    func=run,
    params=(
        Param("kind", "Mask", "choice", "4-neighbor",
              choices=("4-neighbor", "8-neighbor", "diagonal")),
        Param("strength", "Strength c", "float", 1.0, 0, 4, 0.1, 1),
    ),
)
