"""Median filter (order-statistic filter of rank k^2/2)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.types import Operation, Param
from ..core.utils import as_uint8, result, scipy_mode


def run(img, params):
    k = int(params.get("k", 3))
    border = str(params.get("border", "reflect"))
    src = as_uint8(img)
    size = (k, k) if src.ndim == 2 else (k, k, 1)
    out = ndimage.median_filter(src, size=size, mode=scipy_mode(border),
                                cval=0)
    log = (
        "Sort the k x k neighbourhood and keep the middle value.\n"
        f"k = {k}  (rank {k * k // 2 + 1} of {k * k}), border = {border}\n"
        "Unlike a linear average, the median is not pulled far by a single\n"
        "extreme pixel, so it removes salt-and-pepper noise while keeping\n"
        "edges much sharper."
    )
    return result(out, log)


OPERATION = Operation(
    key="median",
    name="Median filter",
    category="Smoothing",
    description=(
        "Middle value of the sorted neighbourhood. Removes "
        "salt-and-pepper noise with less edge blur."
    ),
    func=run,
    params=(
        Param("k", "Kernel size", "int", 3, 3, 15, 2),
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
