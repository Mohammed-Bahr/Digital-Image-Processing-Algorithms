"""Laplacian second-derivative response."""
from __future__ import annotations

from ..core.types import Operation, Param
from ..core.utils import describe_kernel, result
from .common import LAPLACIANS, laplacian_abs


def run(img, params):
    kind = str(params.get("kind", "4-neighbor"))
    if kind not in LAPLACIANS:
        raise ValueError(f"unknown Laplacian mask: {kind}")
    out = laplacian_abs(img, kind)
    log = (
        "Second derivative ∇²f, shown as |∇²f| scaled to 0..255.\n"
        f"Mask: {kind}\n"
        "The coefficients always sum to zero, so a constant region gives\n"
        "zero.  Edges and isolated points give a strong response; the sign\n"
        "distinguishes the bright and dark side of an edge.\n\n"
        + describe_kernel(LAPLACIANS[kind])
    )
    return result(out, log, kernel=LAPLACIANS[kind])


OPERATION = Operation(
    key="laplacian",
    name="Laplacian response",
    category="Sharpening",
    description=(
        "Second derivative |∇²f|. Responds to edges and isolated "
        "points; coefficients always sum to zero."
    ),
    func=run,
    params=(
        Param("kind", "Mask", "choice", "4-neighbor",
              choices=("4-neighbor", "8-neighbor", "diagonal")),
    ),
)
