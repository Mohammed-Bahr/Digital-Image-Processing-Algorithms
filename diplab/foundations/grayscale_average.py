"""RGB -> grayscale with the plain channel average."""
from __future__ import annotations

from ..core.types import Operation
from ..core.utils import result, to_gray


def run(img, params):
    gray = to_gray(img, "average")
    log = (
        "s = (R + G + B) / 3\n"
        "Simple average: equal weight to every channel.  It is easy to\n"
        "compute but does not match human brightness perception."
    )
    return result(gray, log)


OPERATION = Operation(
    key="gray_average",
    name="Grayscale (average)",
    category="Foundations",
    description="RGB -> grayscale with the plain average (R + G + B) / 3.",
    func=run,
)
