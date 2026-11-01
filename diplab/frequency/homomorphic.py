"""Homomorphic filtering: log -> DFT -> H -> IDFT -> exp."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.transforms import (distance_grid, fft2, fftshift2, ifft2,
                               ifftshift2, mask_uint8, padded_size,
                               pad_plane, spectrum_of)
from ..core.utils import EPS, as_float, as_uint8, per_channel, result
from .common import PAD, pad_factor


def run(img, params):
    gl = float(params.get("gamma_l", 0.5))
    gh = float(params.get("gamma_h", 2.0))
    d0 = float(params.get("d0", 30.0))
    pad = pad_factor(params.get("padding", "1"))

    a = as_float(img)
    h, w = a.shape[:2]
    p, q = padded_size(h, w, pad)
    d = distance_grid(p, q)
    d0 = max(d0, EPS)
    H = (gh - gl) * (1.0 - np.exp(-(d * d) / (2.0 * d0 * d0))) + gl

    def one(channel: np.ndarray) -> np.ndarray:
        z = np.log1p(np.maximum(as_float(channel), 0.0))
        F = fft2(pad_plane(z, p, q))
        s = np.real(ifft2(ifftshift2(fftshift2(F) * H))[:h, :w])
        return np.expm1(np.clip(s, -20.0, 20.0))

    out = per_channel(a, one)
    lo, hi = np.percentile(out, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(out.min()), float(out.max())
    if hi > lo:
        out = np.clip((out - lo) / (hi - lo) * 255.0, 0, 255)
    out = as_uint8(out)

    spec = spectrum_of(img, pad)[0]
    log = (
        "f = i · r  ->  log  ->  DFT  ->  H  ->  IDFT  ->  exp  ->  g\n"
        f"γ_L = {gl:g} (illumination gain, < 1 compresses dynamic range),\n"
        f"γ_H = {gh:g} (reflectance gain, > 1 boosts detail), D0 = {d0:g}.\n"
        "H = (γ_H - γ_L)(1 - exp(-D²/2D0²)) + γ_L.\n"
        "The log separates illumination (low frequency) from reflectance\n"
        "(high frequency), so one filter can compress the first and boost\n"
        "the second.  The output is rescaled (2nd..98th percentile) for\n"
        "display."
    )
    return result(out, log,
                  aux={"Input spectrum": spec,
                       "Filter H": mask_uint8(H)})


OPERATION = Operation(
    key="homomorphic",
    name="Homomorphic filtering",
    category="Frequency Domain",
    description=(
        "Compress illumination and boost reflectance at once: "
        "log -> DFT -> H -> IDFT -> exp."
    ),
    func=run,
    params=(
        Param("gamma_l", "γ_L (illumination)", "float", 0.5, 0.05, 1.0, 0.05, 2),
        Param("gamma_h", "γ_H (reflectance)", "float", 2.0, 1.0, 4.0, 0.1, 1),
        Param("d0", "Cutoff D0", "float", 30, 1, 400, 1, 0),
        PAD,
    ),
)
