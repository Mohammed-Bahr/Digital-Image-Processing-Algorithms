"""Maximum filter (rank k^2 order statistic)."""
from __future__ import annotations

from scipy import ndimage

from ..core.types import Operation, Param
from ..core.utils import as_uint8, result, scipy_mode


def run(img, params):
    k = int(params.get("k", 3))
    border = str(params.get("border", "reflect"))
    src = as_uint8(img)
    size = (k, k) if src.ndim == 2 else (k, k, 1)
    out = ndimage.maximum_filter(src, size=size, mode=scipy_mode(border),
                                 cval=0)
    log = (
        "g(x,y) = max{ f(x+i, y+j) } over the k x k window\n"
        f"k = {k}, border = {border}\n"
        "The maximum filter brightens the image: dark isolated points\n"
        "disappear and bright regions grow.  It is the last order\n"
        f"statistic (rank {k * k})."
    )
    return result(out, log)


OPERATION = Operation(
    key="max",
    name="Max filter",
    category="Smoothing",
    description="Largest value in the neighbourhood (rank k²).",
    func=run,
    params=(
        Param("k", "Kernel size", "int", 3, 3, 15, 2),
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
