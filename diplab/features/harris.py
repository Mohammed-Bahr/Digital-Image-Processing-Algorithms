"""Harris corner detector using OpenCV's structure-tensor response."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Operation, Param
from ..core.utils import draw_markers, positive_map, result, to_gray


def harris_response(img, k: float = 0.04, window: int = 3) -> np.ndarray:
    """R = det(M) - k * trace(M)^2 for every pixel."""
    gray = np.ascontiguousarray(to_gray(img).astype(np.float32))
    window = max(3, int(window) | 1)          # odd and >= 3
    return cv2.cornerHarris(gray, blockSize=window, ksize=3, k=float(k))


def harris_corners(img, k: float = 0.04, window: int = 3,
                   threshold_pct: float = 99.0, min_distance: int = 5,
                   max_points: int = 400):
    """Return (list of (row, col) corners, response map)."""
    r = harris_response(img, k=k, window=window)
    if not np.isfinite(r).any():
        return [], r

    local_max = cv2.dilate(r, np.ones((3, 3), dtype=np.float32))
    thr = float(np.percentile(r, threshold_pct))
    candidates = (r >= local_max - 1e-9) & (r > thr) & (r > 0)

    ys, xs = np.nonzero(candidates)
    if len(ys) == 0:
        return [], r

    order = np.argsort(r[ys, xs])[::-1]
    md2 = int(min_distance) ** 2
    points: list[tuple[int, int]] = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if all((y - py) ** 2 + (x - px) ** 2 >= md2 for py, px in points):
            points.append((y, x))
            if len(points) >= int(max_points):
                break
    return points, r


def run(img, params):
    k = float(params.get("k", 0.04))
    window = int(params.get("window", 3))
    pct = float(params.get("threshold", 99.8))
    md = int(params.get("min_distance", 8))
    radius = int(params.get("radius", 5))

    points, r = harris_corners(img, k, window, pct, md)
    vis = draw_markers(img, points, radius=radius)
    log = (
        "R = det(M) - k · trace(M)² = λ1λ2 - k(λ1 + λ2)²\n"
        f"k = {k:g}, window = {window}x{window}, threshold = "
        f"{pct:g}th percentile, min distance = {md} px.\n"
        f"corners found: {len(points)}\n"
        "Corner: both eigenvalues large -> R positive.\n"
        "Edge: one large, one small -> R negative.\n"
        "Flat: both small -> R ≈ 0.\n"
        "The response comes from OpenCV's cornerHarris (Sobel derivatives\n"
        "and a box aggregation window); the non-maximum suppression and\n"
        "minimum-distance thinning are applied here."
    )
    return result(vis, log, aux={"Harris response R": positive_map(r)})


OPERATION = Operation(
    key="harris",
    name="Harris corners",
    category="Features",
    description=(
        "Detect corners: R = det(M) - k·trace(M)². Corners have two "
        "large eigenvalues; edges have one large and one small."
    ),
    func=run,
    params=(
        Param("k", "k", "float", 0.04, 0.01, 0.10, 0.005, 3),
        Param("window", "Window", "int", 3, 3, 11, 2),
        Param("threshold", "Threshold percentile", "float", 99.8, 80, 99.9, 0.1, 1),
        Param("min_distance", "Min distance (px)", "int", 8, 1, 40, 1),
        Param("radius", "Marker radius", "int", 5, 2, 20, 1),
    ),
)
