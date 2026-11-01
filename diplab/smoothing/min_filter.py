"""Minimum filter (rank 1 order statistic)."""
from __future__ import annotations

from scipy import ndimage

from ..core.types import Operation, Param
from ..core.utils import as_uint8, result, scipy_mode


def run(img, params):
    k = int(params.get("k", 3))
    border = str(params.get("border", "reflect"))
    src = as_uint8(img)
    size = (k, k) if src.ndim == 2 else (k, k, 1)
    out = ndimage.minimum_filter(src, size=size, mode=scipy_mode(border),
                                 cval=0)
    log = (
        "g(x,y) = min{ f(x+i, y+j) } over the k x k window\n"
        f"k = {k}, border = {border}\n"
        "The minimum filter darkens the image: bright isolated points\n"
        "disappear, while large bright regions survive shrunken.  It is\n"
        "the first order statistic (rank 1)."
    )
    return result(out, log)


OPERATION = Operation(
    key="min",
    name="Min filter",
    category="Smoothing",
    description="Smallest value in the neighbourhood (rank 1).",
    func=run,
    params=(
        Param("k", "Kernel size", "int", 3, 3, 15, 2),
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
