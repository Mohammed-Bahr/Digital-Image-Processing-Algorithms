"""Adaptive local noise-reduction filter.

f_hat = g - (sigma_eta^2 / sigma_L^2) * (g - m_L)
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Operation, Param
from ..core.utils import (as_float, as_uint8, border_code, per_channel,
                          result)


def local_stats(img, k: int, border: str = "reflect"):
    """Box mean and box variance of a single channel (OpenCV)."""
    ch = np.ascontiguousarray(as_float(img).astype(np.float32))
    mean = cv2.boxFilter(ch, -1, (k, k), normalize=True,
                         borderType=border_code(border))
    mean_sq = cv2.boxFilter(ch * ch, -1, (k, k), normalize=True,
                            borderType=border_code(border))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, var


def run(img, params):
    noise_std = float(params.get("noise_std", 3.0))
    k = int(params.get("k", 3))
    border = str(params.get("border", "reflect"))
    sigma2 = noise_std * noise_std

    def one(channel: np.ndarray) -> np.ndarray:
        ch = as_float(channel)
        mean, var = local_stats(ch, k, border)
        ratio = np.minimum(sigma2 / np.maximum(var, 1e-12), 1.0)
        return ch - ratio * (ch - mean)

    out = as_uint8(per_channel(img, one))

    # Small numeric illustration at the centre pixel of the first channel.
    a = as_float(img)
    first = a if a.ndim == 2 else a[:, :, 0]
    mean, var = local_stats(first, k, border)
    cy, cx = first.shape[0] // 2, first.shape[1] // 2
    m = float(mean[cy, cx])
    v = float(var[cy, cx])
    ratio = min(sigma2 / max(v, 1e-12), 1.0)
    centre = float(first[cy, cx])
    log = (
        "f̂ = g - (σ_η² / σ_L²) · (g - m_L)\n"
        f"σ_η = {noise_std:g} (σ_η² = {sigma2:g}), window k = {k}, "
        f"border = {border}\n"
        "m_L and σ_L² are the local mean and variance in the window.\n"
        "In a flat region σ_L² is small, the ratio approaches 1, and the\n"
        "filter uses the local mean.  On an edge σ_L² is large, the ratio\n"
        "approaches 0, and the original pixel is kept.\n\n"
        f"Centre pixel: g = {centre:.4g}, m_L = {m:.4g}, "
        f"σ_L² = {v:.4g}, ratio = {ratio:.4g}"
    )
    return result(out, log)


OPERATION = Operation(
    key="adaptive",
    name="Adaptive local filter",
    category="Smoothing",
    description=(
        "f̂ = g - (σ_η²/σ_L²)(g - m_L). Smooths flat regions fully but "
        "keeps edges because the local variance is large there."
    ),
    func=run,
    params=(
        Param("noise_std", "Noise σ_η", "float", 3.0, 0, 60, 0.5, 1),
        Param("k", "Window size", "int", 3, 3, 15, 2),
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
