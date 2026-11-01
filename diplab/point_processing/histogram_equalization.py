"""Histogram equalisation through the CDF, exactly as in the notes."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation
from ..core.utils import (L, apply_lut, as_float, histogram_counts,
                          per_channel, result)


def equalization_lut(channel, levels: int = L) -> np.ndarray:
    """Build s_k = round((L - 1) * CDF(k)).

    The notes do not subtract ``cdf_min``; the lowest occupied level can
    therefore map to a value greater than zero.  That is intentional.
    """
    q = np.clip(np.rint(as_float(channel)), 0, levels - 1).astype(np.intp)
    counts = np.bincount(q.ravel(), minlength=levels)[:levels].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.arange(levels, dtype=np.uint8)
    cdf = np.cumsum(counts) / total
    return np.clip(np.rint((levels - 1) * cdf),
                   0, levels - 1).astype(np.uint8)


def equalization_report(img) -> str:
    gray = as_float(img)
    if gray.ndim == 3:
        gray = gray[..., 0]
    lut = equalization_lut(gray)
    counts = histogram_counts(gray)
    used = np.nonzero(counts)[0]
    lines = ["intensity : count -> new intensity"]
    for k in used[:24]:
        lines.append(f"  {int(k):3d} : {int(counts[k]):7d} -> {int(lut[k]):3d}")
    if len(used) > 24:
        lines.append(f"  ... ({len(used) - 24} more levels)")
    return "\n".join(lines)


def run(img, params):
    out = per_channel(img, lambda ch: apply_lut(ch, equalization_lut(ch)))
    log = (
        "s_k = round((L - 1) · CDF(k)),   L = 256\n\n"
        "The CDF is normalised to 0..1, then stretched across the full\n"
        "intensity range.  The output histogram is approximately uniform,\n"
        "so dark or washed-out images gain contrast.\n\n"
        + equalization_report(img)
    )
    return result(out, log)


OPERATION = Operation(
    key="equalize",
    name="Histogram equalization",
    category="Point Processing",
    description=(
        "Remap intensities through the CDF: s_k = round((L-1)·CDF(k)). "
        "Reshapes the histogram towards uniform -> maximum contrast."
    ),
    func=run,
)
