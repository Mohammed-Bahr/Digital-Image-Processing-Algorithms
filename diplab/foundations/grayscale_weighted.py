"""RGB -> grayscale with the perceptual weights from the notes."""
from __future__ import annotations

from ..core.types import Operation
from ..core.utils import result, to_gray

WEIGHTS = (0.30, 0.59, 0.11)


def run(img, params):
    gray = to_gray(img, "weighted")
    log = (
        "s = 0.30 R + 0.59 G + 0.11 B\n"
        "The eye is most sensitive to green, least to blue, so the\n"
        "green channel gets the largest coefficient."
    )
    return result(gray, log)


OPERATION = Operation(
    key="gray_weighted",
    name="Grayscale (weighted)",
    category="Foundations",
    description=(
        "RGB -> grayscale with the perceptual weights "
        "0.30 R + 0.59 G + 0.11 B (the eye is most sensitive to green)."
    ),
    func=run,
)
