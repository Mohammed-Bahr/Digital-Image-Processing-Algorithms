"""Shared numeric helpers.

Only NumPy is used here -- no OpenCV, no scipy.  Convolution/correlation is
built from ``sliding_window_view``, statistics are plain NumPy, and drawing is
done directly on arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

L = 256          # number of gray levels (8-bit)
EPS = 1e-12

# ---------------------------------------------------------------------------
# dtype / channel helpers
# ---------------------------------------------------------------------------

def as_float(img) -> np.ndarray:
    """Return a float64 copy-safe view of *img*."""
    return np.asarray(img, dtype=np.float64)


def as_uint8(img, lo: float = 0, hi: float = 255) -> np.ndarray:
    """Clip + round + cast to uint8 (the standard "clip to valid range" step)."""
    return np.clip(np.rint(img), lo, hi).astype(np.uint8)


def is_color(img) -> bool:
    return getattr(img, "ndim", 0) == 3 and img.shape[2] >= 3


def to_gray(img, method: str = "weighted") -> np.ndarray:
    """RGB -> grayscale.

    ``weighted`` uses the lecture weights 0.3 R + 0.59 G + 0.11 B.
    ``average`` uses (R + G + B) / 3.
    """
    a = as_float(img)
    if a.ndim == 2:
        return a
    if a.shape[2] == 4:
        a = a[..., :3]
    if method == "average":
        return a.mean(axis=2)
    return 0.3 * a[..., 0] + 0.59 * a[..., 1] + 0.11 * a[..., 2]


def per_channel(img, fn):
    """Apply *fn* to a 2-D plane, or to each channel of an RGB image."""
    a = as_float(img)
    if a.ndim == 2:
        return fn(a)
    return np.stack([fn(a[..., i]) for i in range(a.shape[2])], axis=2)


def ensure_rgb(img) -> np.ndarray:
    """uint8 RGB copy (duplicates gray channels)."""
    a = as_uint8(img)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=2)
    if a.shape[2] == 4:
        return a[..., :3].copy()
    return a.copy()


def resize_bilinear(img, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear resize implemented with NumPy (align-corners=False)."""
    a = as_float(img)
    H, W = a.shape[:2]
    out_w, out_h = int(out_w), int(out_h)
    if (W, H) == (out_w, out_h):
        return a.copy()

    ys = (np.arange(out_h) + 0.5) * (H / out_h) - 0.5
    xs = (np.arange(out_w) + 0.5) * (W / out_w) - 0.5
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    if a.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]

    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


# ---------------------------------------------------------------------------
# padding + neighbourhood access
# ---------------------------------------------------------------------------

_PAD_ALIASES = {
    "zeros": "constant",
    "zero": "constant",
    "replicate": "edge",
    "edge": "edge",
    "reflect": "symmetric",     # matches the note's 3 2 1 | 1 2 3 | 3 2 1
    "symmetric": "symmetric",
    "wrap": "wrap",
}


def pad_image(img, ph: int, pw: int, mode: str = "reflect") -> np.ndarray:
    """Pad a 2-D or 3-D (H, W, C) array by (ph, pw)."""
    m = _PAD_ALIASES.get(mode, mode)
    if img.ndim == 2:
        pads = ((ph, ph), (pw, pw))
    else:
        pads = ((ph, ph), (pw, pw), (0, 0))
    if m == "constant":
        return np.pad(img, pads, mode=m, constant_values=0)
    return np.pad(img, pads, mode=m)


def windows(arr, kh: int, kw: int) -> np.ndarray:
    """All kh x kw neighbourhoods, shape (H-kh+1, W-kw+1, kh, kw)."""
    return sliding_window_view(arr, (kh, kw))


def correlate(img, kernel, border: str = "reflect") -> np.ndarray:
    """Spatial correlation:  g(x, y) = sum_{s,t} w(s, t) f(x+s, y+t).

    This is exactly the formula in the lecture notes (also called "masking").
    Kernels must have odd size so the centre is well defined.
    """
    k = as_float(kernel)
    if k.ndim != 2:
        raise ValueError("kernel must be 2-D")
    kh, kw = k.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel size must be odd")
    ph, pw = kh // 2, kw // 2

    def one(ch):
        padded = pad_image(ch, ph, pw, border)
        win = windows(padded, kh, kw)
        return np.einsum("ijkl,kl->ij", win, k)

    return per_channel(img, one)


def box_filter(img, k: int, border: str = "reflect") -> np.ndarray:
    """Uniform k x k mean filter using an integral image -- O(HW) per pixel."""
    if k % 2 == 0:
        raise ValueError("kernel size must be odd")
    r = k // 2
    denom = float(k * k)

    def one(ch):
        p = pad_image(ch, r, r, border)
        ii = p.cumsum(axis=0).cumsum(axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
        s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
        return s / denom

    return per_channel(img, one)


def sliding_reduce(img, k: int, reducer, border: str = "reflect",
                   block_rows: int = 0) -> np.ndarray:
    """Evaluate ``reducer`` on every k x k neighbourhood.

    Windows are materialised in row blocks so memory stays bounded even for
    large images and large kernels.
    """
    if k % 2 == 0:
        raise ValueError("kernel size must be odd")
    r = k // 2
    a = as_float(img)
    H, W = a.shape[:2]
    if block_rows <= 0:
        block_rows = int(np.clip(24_000_000 // max(1, W * k * k), 8, 128))

    def one(ch):
        p = pad_image(ch, r, r, border)
        out = np.empty(ch.shape, dtype=np.float64)
        for y0 in range(0, H, block_rows):
            y1 = min(H, y0 + block_rows)
            sub = p[y0:y1 + k - 1, :]
            win = windows(sub, k, k)
            win = win.reshape(win.shape[0], win.shape[1], k * k)
            out[y0:y1] = reducer(win)
        return out

    return per_channel(a, one)


def gaussian_kernel_2d(size: int, sigma: float | None = None) -> np.ndarray:
    """Normalised 2-D Gaussian mask (used as the Harris window function)."""
    size = int(size)
    if size % 2 == 0:
        size += 1
    if sigma is None:
        sigma = max(size / 3.0, 0.8)
    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return g / g.sum()


def maximum_filter(a, size: int = 3) -> np.ndarray:
    """Sliding maximum (used for Harris non-maximum suppression)."""
    r = size // 2
    p = np.pad(as_float(a), r, mode="constant", constant_values=-np.inf)
    win = windows(p, size, size)
    return win.max(axis=(2, 3))


# ---------------------------------------------------------------------------
# histograms / scaling
# ---------------------------------------------------------------------------

def histogram_counts(img, bins: int = 256) -> np.ndarray:
    """Count pixels per intensity (uses luminance for colour images)."""
    g = to_gray(img) if is_color(img) else as_float(img)
    q = np.clip(np.rint(g), 0, bins - 1).astype(np.intp)
    return np.bincount(q.ravel(), minlength=bins)[:bins].astype(np.float64)


def cdf_from_counts(counts) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0:
        return np.zeros_like(c)
    return np.cumsum(c) / total


def normalize_to_uint8(x, lo: float | None = None,
                       hi: float | None = None) -> np.ndarray:
    """Linearly rescale an arbitrary array into 0..255 uint8."""
    a = as_float(x)
    if lo is None:
        lo = float(np.min(a))
    if hi is None:
        hi = float(np.max(a))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return as_uint8((a - lo) / (hi - lo) * 255.0)


def stretch_minmax(img, lo_pct: float = 0, hi_pct: float = 100) -> np.ndarray:
    """Contrast stretch with optional percentile clipping."""
    a = as_float(img)
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi <= lo:
        return as_uint8(a)
    return as_uint8((a - lo) / (hi - lo) * 255.0)


# ---------------------------------------------------------------------------
# drawing (corners / markers)
# ---------------------------------------------------------------------------

def draw_markers(img, points, radius: int = 5,
                 color=(255, 55, 55), ring=(255, 255, 255)) -> np.ndarray:
    """Draw hollow circles around (row, col) points on an RGB uint8 image."""
    out = ensure_rgb(img)
    H, W = out.shape[:2]
    r = int(radius)
    for (y, x) in points:
        y0, y1 = max(0, y - r - 1), min(H, y + r + 2)
        x0, x1 = max(0, x - r - 1), min(W, x + r + 2)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        sub = out[y0:y1, x0:x1]
        sub[d2 <= r * r] = color
        sub[(d2 <= (r + 1) ** 2) & (d2 > r * r)] = ring
    return out


def draw_line(arr, y0: float, x0: float, y1: float, x1: float,
              color=(255, 255, 255), thickness: int = 1):
    """Draw a line into an HxW or HxWxC uint8 array (in place)."""
    H, W = arr.shape[:2]
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 2
    ts = np.linspace(0.0, 1.0, n)
    ys = np.rint(y0 + (y1 - y0) * ts).astype(np.intp)
    xs = np.rint(x0 + (x1 - x0) * ts).astype(np.intp)
    offs = range(-(thickness // 2), thickness // 2 + 1)
    for dy in offs:
        for dx in offs:
            yy = np.clip(ys + dy, 0, H - 1)
            xx = np.clip(xs + dx, 0, W - 1)
            arr[yy, xx] = color
    return arr


# ---------------------------------------------------------------------------
# small formatting helpers used by the GUI log
# ---------------------------------------------------------------------------

def format_matrix(m, decimals: int = 2) -> str:
    a = as_float(m)
    if a.ndim == 1:
        a = a[None, :]
    cells = []
    for row in a:
        cells.append([(f"{v:.{decimals}f}".rstrip("0").rstrip(".") or "0")
                      for v in row])
    width = max(len(c) for row in cells for c in row)
    return "\n".join("  " + "  ".join(c.rjust(width) for c in row)
                     for row in cells)


def describe_kernel(k) -> str:
    a = as_float(k)
    s = float(a.sum())
    if a.size > 25:
        return f"kernel {a.shape[0]}x{a.shape[1]}, sum = {s:g}"
    flat = [f"{v:g}" for v in a.ravel()]
    return (f"kernel {a.shape[0]}x{a.shape[1]}, sum = {s:g}, "
            f"coefficients = [{', '.join(flat)}]")


def next_power_of_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
