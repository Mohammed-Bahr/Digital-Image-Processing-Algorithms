"""Shared frequency-domain masks and helpers."""
from __future__ import annotations

import numpy as np

from ..core.types import Param
from ..core.transforms import (apply_filter, distance_grid, filter_aux,
                               kept_percent)
from ..core.utils import EPS, result

PAD = Param("padding", "FFT padding", "choice", "1",
            choices=("1", "2"))

KIND = Param("kind", "Filter type", "choice", "butterworth",
             choices=("ideal", "butterworth", "gaussian"))


def pad_factor(value) -> float:
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 1.0


def lowpass_mask(shape, d0: float, n: float = 2.0,
                 kind: str = "ideal") -> np.ndarray:
    """Ideal / Butterworth / Gaussian low-pass H(u, v)."""
    D = distance_grid(*shape)
    d0 = max(float(d0), EPS)
    if kind == "ideal":
        return (D <= d0).astype(np.float64)
    if kind == "butterworth":
        return 1.0 / (1.0 + (D / d0) ** (2.0 * max(float(n), 0.01)))
    if kind == "gaussian":
        return np.exp(-(D * D) / (2.0 * d0 * d0))
    raise ValueError(f"unknown low-pass kind: {kind}")


def highpass_mask(shape, d0: float, n: float = 2.0,
                  kind: str = "ideal") -> np.ndarray:
    """A high-pass is the complement of its low-pass partner."""
    return 1.0 - lowpass_mask(shape, d0, n, kind)


def band_mask(shape, d0: float, d1: float, n: float = 2.0,
              kind: str = "ideal", reject: bool = False) -> np.ndarray:
    """Band-pass = low-pass(D1) * high-pass(D0); reject complements it."""
    lo, hi = sorted((max(float(d0), EPS), max(float(d1), EPS)))
    bp = (lowpass_mask(shape, hi, n, kind)
          * (1.0 - lowpass_mask(shape, lo, n, kind)))
    return 1.0 - bp if reject else bp


def notch_reject_mask(shape, du: float, dv: float,
                      radius: float) -> np.ndarray:
    """Zero two symmetric discs around (du, dv) and (-du, -dv)."""
    p, q = shape
    yy = np.arange(p)[:, None] - p // 2
    xx = np.arange(q)[None, :] - q // 2
    d1 = np.hypot(yy - float(dv), xx - float(du))
    d2 = np.hypot(yy + float(dv), xx + float(du))
    r = max(float(radius), EPS)
    mask = np.ones(shape, dtype=np.float64)
    mask[d1 <= r] = 0.0
    mask[d2 <= r] = 0.0
    return mask


def run_filter(img, mask_fn, pad, log: str, extra_aux=None):
    """Apply a mask and wrap the result with spectra tabs."""
    out, mask, spec_in, spec_out = apply_filter(img, mask_fn, pad_factor(pad))
    aux = filter_aux(mask, spec_in, spec_out)
    if extra_aux:
        aux.update(extra_aux)
    kept = kept_percent(mask)
    full_log = log + f"\n\nMask mean = {kept:.1f}% (fraction of the spectrum kept)."
    return result(out, full_log, aux=aux)
