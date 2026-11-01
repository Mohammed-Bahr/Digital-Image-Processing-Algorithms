"""The lecture's weighted 3x3 average (1/16 kernel)."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import (as_uint8, correlate, describe_kernel, result)

WEIGHTED_3X3 = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], dtype=np.float64) / 16.0


def run(img, params):
    border = str(params.get("border", "reflect"))
    out = as_uint8(correlate(img, WEIGHTED_3X3, border))
    log = (
        "g = (1/16) · [[1,2,1],[2,4,2],[1,2,1]] ⊛ f\n"
        f"border = {border}\n"
        "The centre pixel counts four times as much as an edge neighbour\n"
        "and sixteen times as much as a corner.  Because the weights fall\n"
        "off smoothly, this blurs less than a uniform 3x3 average.\n\n"
        + describe_kernel(WEIGHTED_3X3)
    )
    return result(out, log, kernel=WEIGHTED_3X3)


OPERATION = Operation(
    key="weighted",
    name="Weighted average 1/16",
    category="Smoothing",
    description=(
        "(1/16)·[[1,2,1],[2,4,2],[1,2,1]]. The centre counts 4x, so it "
        "blurs less than a uniform average."
    ),
    func=run,
    params=(
        Param("border", "Border padding", "choice", "reflect",
              choices=("reflect", "replicate", "zeros")),
    ),
)
