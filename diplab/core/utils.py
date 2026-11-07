"""Shared helpers: type conversion, colour, borders, LUTs, drawing."""
from __future__ import annotations

import cv2
import numpy as np

from .types import OpResult

L = 256
EPS = 1e-12

# cv2.BORDER_REFLECT is the symmetric padding used in the notes:
#   fedcba | abcdefgh | hgfedcb
BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "symmetric": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "edge": cv2.BORDER_REPLICATE,
    "zeros": cv2.BORDER_CONSTANT,
    "constant": cv2.BORDER_CONSTANT,
    "wrap": cv2.BORDER_WRAP,
}

SCIPY_MODES = {
    "reflect": "reflect",
    "symmetric": "reflect",
    "replicate": "nearest",
    "edge": "nearest",
    "zeros": "constant",
    "constant": "constant",
    "wrap": "wrap",
}


def as_float(img: np.ndarray) -> np.ndarray:
    """Return a float64 view/copy of an image."""
    return np.asarray(img, dtype=np.float64)


def as_uint8(img: np.ndarray, lo: float = 0, hi: float = 255) -> np.ndarray:
    """Round, clip and cast an image to uint8."""
    return np.clip(np.rint(img), lo, hi).astype(np.uint8) 

# means constraining pixel values to a valid 0–255 
# range so that out-of-bounds or floating-point numbers do not wrap around or 
# break when converted to an 8-bit unsigned integer


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel uint8 RGB image."""
    a = as_uint8(img)
    if a.ndim == 2:
        return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    if a.shape[2] == 1:
        return cv2.cvtColor(a[:, :, 0], cv2.COLOR_GRAY2RGB)
    if a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
    return a.copy()


def to_gray(img: np.ndarray, method: str = "weighted") -> np.ndarray:
    """Convert to a float64 grayscale plane.

    ``weighted`` uses the lecture weights 0.30 R + 0.59 G + 0.11 B;
    ``average`` uses the plain mean.
    """
    a = as_float(img)
    if a.ndim == 2:
        return a
    if a.shape[2] == 4:
        a = a[:, :, :3]
    if method == "average":
        return a.mean(axis=2)
    return 0.30 * a[:, :, 0] + 0.59 * a[:, :, 1] + 0.11 * a[:, :, 2]


def per_channel(img: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn`` to each colour channel independently."""
    a = as_float(img)
    if a.ndim == 2:
        return fn(a)
    return np.stack([fn(a[:, :, c]) for c in range(a.shape[2])], axis=2)


def apply_lut(img: np.ndarray, lut) -> np.ndarray:
    """Apply a 256-entry lookup table with OpenCV."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = as_uint8(a)
    a = np.ascontiguousarray(a)
    table = np.ascontiguousarray(np.asarray(lut, dtype=np.uint8))
    return cv2.LUT(a, table)


def resize(img: np.ndarray, out_w: int, out_h: int,
           interpolation: str = "linear") -> np.ndarray:
    """Resize with OpenCV (area is best for shrinking)."""
    modes = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }
    return cv2.resize(img, (int(out_w), int(out_h)),
                      interpolation=modes.get(interpolation, cv2.INTER_LINEAR))


def border_code(mode: str) -> int:
    return BORDER_MODES.get(mode, cv2.BORDER_REFLECT)


def scipy_mode(mode: str) -> str:
    return SCIPY_MODES.get(mode, "reflect")


def correlate(img: np.ndarray, kernel, border: str = "reflect") -> np.ndarray:
    """2-D correlation (sum of w * f), matching the notes' convention."""
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("correlation kernel must be 2-D with odd dimensions")
    src = as_float(img)
    return cv2.filter2D(src, -1, k, borderType=border_code(border))


def histogram_counts(channel: np.ndarray, levels: int = L) -> np.ndarray:
    """Integer histogram of a single channel."""
    q = np.clip(np.rint(as_float(channel)), 0, levels - 1).astype(np.intp)
    counts = np.bincount(q.ravel(), minlength=levels)
    return counts[:levels]


def normalize_to_uint8(x: np.ndarray, low: float = 0.0,
                       high: float = 255.0) -> np.ndarray:
    """Min-max stretch to [low, high] and cast to uint8."""
    a = as_float(x)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < EPS:
        return np.full(a.shape, int(round((low + high) / 2)), dtype=np.uint8)
    return as_uint8((a - lo) * (high - low) / (hi - lo) + low)


def positive_map(x: np.ndarray) -> np.ndarray:
    """Show only non-negative values, stretched to 255."""
    a = np.where(np.isfinite(x), x, 0.0)
    peak = float(a.max())
    if peak <= 0:
        return np.zeros(a.shape, dtype=np.uint8)
    return as_uint8(np.clip(a, 0, peak) / peak * 255.0)


def draw_markers(img: np.ndarray, points, radius: int = 5,
                 color=(255, 60, 60), ring=(255, 255, 255)) -> np.ndarray:
    """Draw small filled circles on a copy of an RGB image."""
    out = ensure_rgb(img)
    r = max(1, int(radius))
    for y, x in points:
        centre = (int(round(x)), int(round(y)))
        cv2.circle(out, centre, r, tuple(int(c) for c in color),
                   -1, cv2.LINE_AA)
        cv2.circle(out, centre, r + 1, tuple(int(c) for c in ring),
                   1, cv2.LINE_AA)
    return out


def format_matrix(kernel: np.ndarray, digits: int = 3) -> str:
    """Plain-text matrix with aligned columns."""
    a = np.asarray(kernel, dtype=np.float64)
    cells = [[f"{v:.{digits}g}" for v in row] for row in a]
    widths = [max(len(cells[r][c]) for r in range(a.shape[0]))
              for c in range(a.shape[1])]
    lines = ["  ".join(cells[r][c].rjust(widths[c])
                       for c in range(a.shape[1]))
             for r in range(a.shape[0])]
    return "\n".join("[ " + line + " ]" for line in lines) if a.shape[0] > 1 \
        else "[ " + lines[0] + " ]"


def describe_kernel(kernel: np.ndarray) -> str:
    a = np.asarray(kernel, dtype=np.float64)
    return (f"Filter mask {a.shape[0]}x{a.shape[1]} "
            f"(sum {a.sum():.4g}):\n{format_matrix(a)}")


def next_power_of_two(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def result(image: np.ndarray, log: str = "", kernel=None,
           aux=None, **meta) -> OpResult:
    """Small constructor used by every algorithm module."""
    return OpResult(as_uint8(image), log=log, kernel=kernel,
                    aux=dict(aux or {}), meta=dict(meta))
