"""Point processing + histogram methods -- notes 00 (Lectures 1-2).

Every operator works pixel-by-pixel on intensity:  s = T(r).  Because of
that, all of them can be evaluated through a 256-entry lookup table (LUT).
"""

from __future__ import annotations

import numpy as np

from .utils import (L, as_float, as_uint8, histogram_counts, per_channel)


# ---------------------------------------------------------------------------
# negative
# ---------------------------------------------------------------------------

def negative(img) -> np.ndarray:
    """s = (L - 1) - r   -> flips dark/light."""
    return as_uint8(255.0 - as_float(img))


# ---------------------------------------------------------------------------
# logarithmic:  s = c * log(1 + r)
# ---------------------------------------------------------------------------

def log_transform(img, c: float | None = None) -> np.ndarray:
    """Expand dark values, compress bright values.

    ``c`` defaults to 255 / log(1 + r_max) so the output fills 0..255.
    """
    a = as_float(img)
    if c is None or c <= 0:
        r_max = float(a.max())
        c = 255.0 / np.log1p(r_max) if r_max > 0 else 1.0
    return as_uint8(c * np.log1p(np.maximum(a, 0.0)))


# ---------------------------------------------------------------------------
# gamma:  s = c * r^gamma   (r normalised to 0..1 first)
# ---------------------------------------------------------------------------

def gamma_transform(img, gamma: float, c: float = 1.0) -> np.ndarray:
    """gamma < 1 brightens, gamma > 1 darkens."""
    r = np.clip(as_float(img) / 255.0, 0.0, 1.0)
    return as_uint8(255.0 * c * np.power(r, max(gamma, 1e-6)))


# ---------------------------------------------------------------------------
# piecewise linear
# ---------------------------------------------------------------------------

def piecewise_linear(img, r1: float, s1: float,
                     r2: float, s2: float, levels: int = L) -> np.ndarray:
    """Three-segment control-point mapping from the notes.

    Segments: (0,0)->(r1,s1), (r1,s1)->(r2,s2), (r2,s2)->(L-1,L-1).
    """
    if not (1 <= r1 < r2 <= levels - 2):
        raise ValueError("control points need 1 <= r1 < r2 <= 254")
    if not (0 <= s1 <= 255 and 0 <= s2 <= 255):
        raise ValueError("s1 and s2 must be in 0..255")

    a = as_float(img)
    out = np.empty_like(a)

    m0 = a <= r1
    out[m0] = (s1 / r1) * a[m0]

    m1 = (a > r1) & (a <= r2)
    out[m1] = ((s2 - s1) / (r2 - r1)) * (a[m1] - r1) + s1

    m2 = a > r2
    out[m2] = (((levels - 1) - s2) / ((levels - 1) - r2)) * (a[m2] - r2) + s2

    return as_uint8(out)


# ---------------------------------------------------------------------------
# contrast stretching (min-max normalisation)
# ---------------------------------------------------------------------------

def contrast_stretch(img, low_pct: float = 0.0,
                     high_pct: float = 100.0) -> np.ndarray:
    """Map [x_min, x_max] -> [0, 255].

    Percentiles other than 0/100 clip outliers (the weakness of the plain
    min-max version, as noted in the lecture).
    """
    a = as_float(img)
    lo = float(np.percentile(a, low_pct))
    hi = float(np.percentile(a, high_pct))
    if hi <= lo:
        return as_uint8(a)
    return as_uint8((a - lo) / (hi - lo) * 255.0)


# ---------------------------------------------------------------------------
# histogram equalisation
# ---------------------------------------------------------------------------

def equalization_lut(channel, levels: int = L) -> np.ndarray:
    """Build the CDF mapping table  s_k = round((L-1) * CDF(k)).

    This follows the notes exactly (no cdf_min subtraction), which is why
    the lowest occupied level can map to a value > 0.
    """
    q = np.clip(np.rint(as_float(channel)), 0, levels - 1).astype(np.intp)
    counts = np.bincount(q.ravel(), minlength=levels)[:levels].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.arange(levels, dtype=np.uint8)
    cdf = np.cumsum(counts) / total
    lut = np.clip(np.rint((levels - 1) * cdf), 0, levels - 1)
    return lut.astype(np.uint8)


def apply_lut(img, lut) -> np.ndarray:
    """Replace every intensity by lut[intensity] (vectorised lookup)."""
    a = np.clip(np.rint(as_float(img)), 0, 255).astype(np.intp)
    lut = np.asarray(lut, dtype=np.uint8)
    return lut[a]


def histogram_equalization(img) -> np.ndarray:
    """Equalise each channel through its own CDF-derived LUT."""
    return per_channel(img, lambda ch: apply_lut(ch, equalization_lut(ch)))


def equalization_report(img) -> str:
    """Small textual table (old -> new) for the log panel."""
    g = as_float(img)
    if g.ndim == 3:
        g = g[..., 0]
    lut = equalization_lut(g)
    counts = histogram_counts(g)
    used = np.nonzero(counts)[0]
    lines = ["intensity : count -> new intensity"]
    for k in used[:24]:
        lines.append(f"  {int(k):3d} : {int(counts[k]):7d} -> {int(lut[k]):3d}")
    if len(used) > 24:
        lines.append(f"  ... ({len(used) - 24} more levels)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers used by the GUI's transfer-curve panel
# ---------------------------------------------------------------------------

def lut_curve(fn, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a point operation on the 0..255 ramp, for plotting."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    y = fn(ramp, **kwargs)
    y = np.asarray(y).reshape(-1)
    return np.arange(256), y.astype(np.float64)
