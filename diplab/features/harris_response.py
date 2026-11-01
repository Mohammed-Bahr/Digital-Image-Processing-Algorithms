"""Harris response map."""
from __future__ import annotations

from ..core.types import Operation, Param
from ..core.utils import positive_map, result
from .harris import harris_response


def run(img, params):
    k = float(params.get("k", 0.04))
    window = int(params.get("window", 3))
    r = harris_response(img, k, window)
    log = (
        "Harris response R = det(M) - k·trace(M)², positive part "
        "rescaled to 0..255.\n"
        f"k = {k:g}, window = {window}x{window}.\n"
        "Bright pixels are corner candidates; dark pixels are flat or\n"
        "edge-like (R is negative on an edge)."
    )
    return result(positive_map(r), log, meta={"response": r})


OPERATION = Operation(
    key="harris_map",
    name="Harris response map",
    category="Features",
    description="The raw R map: bright pixels are corner candidates.",
    func=run,
    params=(
        Param("k", "k", "float", 0.04, 0.01, 0.10, 0.005, 3),
        Param("window", "Window", "int", 3, 3, 11, 2),
    ),
)
