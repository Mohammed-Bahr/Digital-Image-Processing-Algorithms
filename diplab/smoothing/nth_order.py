"""Arbitrary order-statistic (rank) filter."""
from __future__ import annotations

from scipy import ndimage

from ..core.types import Operation, Param
from ..core.utils import as_uint8, result, scipy_mode


def run(img, params):
    k = int(params.get("k", 3))
    rank = int(params.get("rank", 5))
    border = str(params.get("border", "reflect"))
    max_rank = k * k
    rank = max(1, min(rank, max_rank))
    src = as_uint8(img)
    size = (k, k) if src.ndim == 2 else (k, k, 1)
    # scipy's rank is 0-based; the GUI uses 1-based ranks.
    out = ndimage.rank_filter(src, rank=rank - 1, size=size,
                              mode=scipy_mode(border), cval=0)
    if rank == 1:
        name = "minimum"
    elif rank == max_rank:
        name = "maximum"
    elif rank == max_rank // 2 + 1:
        name = "median"
    else:
        name = "order statistic"
    log = (
        "Sort the k x k window and take the n-th value.\n"
        f"k = {k}, n = {rank} of {max_rank} (1 = min, "
        f"{max_rank // 2 + 1} = median, {max_rank} = max)\n"
        f"border = {border}\n"
        f"This is the {name} filter: small ranks suppress bright spots,\n"
        "large ranks suppress dark spots, the median removes impulses\n"
        "without moving edges."
    )
    return result(out, log)


OPERATION = Operation(
    key="nth",
    name="Nth-order statistic",
    category="Smoothing",
    description=(
        "Pick the n-th value after sorting the neighbourhood "
        "(1 = min, middle = median, k² = max)."
    ),
    func=run,
    params=(
        Param("k", "Kernel size", "int", 3, 3, 15, 2),
        Param("rank", "Rank (1..k²)", "int", 5, 1, 225, 1),
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
