"""Uniform k x k average (box filter)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Operation, Param
from ..core.utils import (as_uint8, border_code, describe_kernel, result)


def run(img, params):
    k = int(params.get("k", 3))
    border = str(params.get("border", "reflect"))
    src = np.ascontiguousarray(as_uint8(img))
    out = cv2.blur(src, (k, k), borderType=border_code(border))
    kernel = np.ones((k, k), dtype=np.float64) / (k * k)
    log = (
        "g(x,y) = (1/k²) · Σ f(x+i, y+j)   over the k x k window\n"
        f"k = {k}, border = {border}\n"
        "Every neighbour has the same weight, the coefficients sum to 1,\n"
        "so the average brightness is preserved.  A larger kernel removes\n"
        "more noise but blurs edges more.\n\n"
        + describe_kernel(kernel)
    )
    return result(out, log, kernel=kernel)


OPERATION = Operation(
    key="box",
    name="Average (box) filter",
    category="Smoothing",
    description=(
        "Uniform k x k average. Sum of coefficients = 1, so brightness "
        "is preserved. Bigger kernel = more denoising and more blur."
    ),
    func=run,
    params=(
        Param("k", "Kernel size", "int", 3, 3, 31, 2),
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
