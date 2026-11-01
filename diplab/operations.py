"""Operation registry.

Every algorithm lives in its own module under a topic folder.  Each module
exposes one ``OPERATION`` object; this file simply collects them in the
order they appear in the notes.
"""
from __future__ import annotations

from .core.types import OpResult, Operation, Param
from .foundations import (extract_channel, grayscale_average,
                          grayscale_weighted)
from .point_processing import (contrast_stretching, gamma,
                               histogram_equalization, logarithmic, negative,
                               piecewise_linear)
from .noise import gaussian_noise, salt_and_pepper
from .smoothing import (adaptive_local, box_filter, max_filter, median_filter,
                        min_filter, nth_order, weighted_average)
from .sharpening import (laplacian, laplacian_sharpening, sobel,
                         sobel_orientation)
from .frequency import (bandpass, bandreject, highpass, homomorphic, lowpass,
                        notch, spectrum)
from .features import harris, harris_response, hog

CATEGORY_ORDER = [
    "Foundations",
    "Point Processing",
    "Noise",
    "Smoothing",
    "Sharpening",
    "Frequency Domain",
    "Features",
]

OPERATIONS: tuple[Operation, ...] = (
    # ---- foundations ------------------------------------------------------
    grayscale_weighted.OPERATION,
    grayscale_average.OPERATION,
    extract_channel.OPERATION,
    # ---- point processing -------------------------------------------------
    negative.OPERATION,
    logarithmic.OPERATION,
    gamma.OPERATION,
    piecewise_linear.OPERATION,
    contrast_stretching.OPERATION,
    histogram_equalization.OPERATION,
    # ---- noise ------------------------------------------------------------
    salt_and_pepper.OPERATION,
    gaussian_noise.OPERATION,
    # ---- smoothing --------------------------------------------------------
    box_filter.OPERATION,
    weighted_average.OPERATION,
    median_filter.OPERATION,
    min_filter.OPERATION,
    max_filter.OPERATION,
    nth_order.OPERATION,
    adaptive_local.OPERATION,
    # ---- sharpening -------------------------------------------------------
    sobel.OPERATION,
    sobel_orientation.OPERATION,
    laplacian.OPERATION,
    laplacian_sharpening.OPERATION,
    # ---- frequency --------------------------------------------------------
    spectrum.OPERATION,
    lowpass.OPERATION,
    highpass.OPERATION,
    bandpass.OPERATION,
    bandreject.OPERATION,
    notch.OPERATION,
    homomorphic.OPERATION,
    # ---- features ---------------------------------------------------------
    harris.OPERATION,
    harris_response.OPERATION,
    hog.OPERATION,
)


def operations_by_category() -> dict[str, list[Operation]]:
    grouped: dict[str, list[Operation]] = {c: [] for c in CATEGORY_ORDER}
    for op in OPERATIONS:
        grouped.setdefault(op.category, []).append(op)
    return grouped


def find_operation(key: str) -> Operation | None:
    for op in OPERATIONS:
        if op.key == key:
            return op
    return None


__all__ = [
    "CATEGORY_ORDER", "OPERATIONS", "OpResult", "Operation", "Param",
    "find_operation", "operations_by_category",
]
