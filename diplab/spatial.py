"""Spatial filtering -- notes 01 (Lecture 3).

Linear masks (average / weighted), order-statistic filters (min, max, median,
nth), the adaptive local noise-reduction filter, and the derivative operators
(Sobel, Scharr, Laplacian).
"""

from __future__ import annotations

import numpy as np

from .utils import (as_float, as_uint8, box_filter, correlate, per_channel,
                    sliding_reduce, to_gray)

# ---------------------------------------------------------------------------
# kernels from the notes
# ---------------------------------------------------------------------------

BOX3 = np.ones((3, 3), dtype=np.float64) / 9.0

WEIGHTED3 = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=np.float64) / 16.0

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)

SCHARR_X = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]], dtype=np.float64)

SCHARR_Y = np.array([[-3, -10, -3],
                     [0, 0, 0],
                     [3, 10, 3]], dtype=np.float64)

LAPLACIANS = {
    "4-neighbor": np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=np.float64),
    "8-neighbor": np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]], dtype=np.float64),
    "diagonal": np.array([[-1, 2, -1],
                          [2, -4, 2],
                          [-1, 2, -1]], dtype=np.float64),
}

KERNELS = {
    "box-3": BOX3,
    "weighted-3": WEIGHTED3,
    "sobel-x": SOBEL_X,
    "sobel-y": SOBEL_Y,
    "scharr-x": SCHARR_X,
    "scharr-y": SCHARR_Y,
    **{f"laplacian-{k}": v for k, v in LAPLACIANS.items()},
}


# ---------------------------------------------------------------------------
# linear smoothing
# ---------------------------------------------------------------------------

def box_average(img, k: int = 3, border: str = "reflect"):
    """Uniform average: every coefficient is 1/k^2 (sum = 1 -> brightness kept)."""
    return as_uint8(box_filter(img, k, border))


def weighted_average(img, border: str = "reflect"):
    """Classic 1/16 mask: centre 4, orthogonal 2, diagonal 1."""
    return as_uint8(correlate(img, WEIGHTED3, border))


# ---------------------------------------------------------------------------
# order-statistic (non-linear) filters
# ---------------------------------------------------------------------------

def median_filter(img, k: int = 3, border: str = "reflect"):
    """Middle value of the sorted neighbourhood -- great for salt & pepper."""
    return as_uint8(sliding_reduce(img, k, lambda w: np.median(w, axis=-1), border))


def min_filter(img, k: int = 3, border: str = "reflect"):
    """Smallest value -> darkens bright specks (erosion-like)."""
    return as_uint8(sliding_reduce(img, k, lambda w: w.min(axis=-1), border))


def max_filter(img, k: int = 3, border: str = "reflect"):
    """Largest value -> removes dark specks (dilation-like)."""
    return as_uint8(sliding_reduce(img, k, lambda w: w.max(axis=-1), border))


def nth_order_filter(img, k: int = 3, rank: int = 5,
                     border: str = "reflect"):
    """Pick the rank-th value (1 = min, k^2 = max, middle = median)."""
    if k % 2 == 0:
        raise ValueError("kernel size must be odd")
    total = k * k
    rank = int(np.clip(rank, 1, total))
    idx = rank - 1

    def pick(w):
        return np.partition(w, idx, axis=-1)[..., idx]

    return as_uint8(sliding_reduce(img, k, pick, border))


# ---------------------------------------------------------------------------
# adaptive local noise reduction
# ---------------------------------------------------------------------------

def adaptive_local(img, noise_std: float = 3.0, k: int = 3,
                   border: str = "reflect"):
    """Adaptive filter from the notes.

        f_hat = g - (sigma_eta^2 / sigma_L^2) * (g - m_L)

    ``noise_std`` is sigma_eta (standard deviation of the noise).
    The ratio is clamped at 1, as the lecture requires.
    """
    sigma_eta2 = max(float(noise_std), 0.0) ** 2

    def one(ch):
        m_l = box_filter(ch, k, border)
        mean_sq = box_filter(ch * ch, k, border)
        var_l = np.maximum(mean_sq - m_l * m_l, 0.0)
        if sigma_eta2 <= 0:
            return ch.copy()
        ratio = np.minimum(sigma_eta2 / np.maximum(var_l, 1e-12), 1.0)
        return ch - ratio * (ch - m_l)

    return as_uint8(per_channel(img, one))


def adaptive_stats(img, noise_std: float = 3.0, k: int = 3,
                   y: int | None = None, x: int | None = None) -> str:
    """Explain the adaptive result for one pixel (top-left by default)."""
    g = to_gray(img)
    H, W = g.shape
    y = 0 if y is None else int(np.clip(y, 0, H - 1))
    x = 0 if x is None else int(np.clip(x, 0, W - 1))
    r = k // 2
    p = np.pad(g, r, mode="symmetric")
    win = p[y:y + k, x:x + k]
    m_l = float(win.mean())
    var_l = float(win.var())
    ratio = min((noise_std ** 2) / var_l, 1.0) if var_l > 0 else 0.0
    out = float(g[y, x] - ratio * (g[y, x] - m_l))
    return (f"sample pixel ({y}, {x}): g = {g[y, x]:.1f}, "
            f"m_L = {m_l:.2f}, sigma_L^2 = {var_l:.2f}, "
            f"sigma_eta^2 = {noise_std ** 2:.2f}, ratio = {ratio:.2f} "
            f"-> f_hat = {out:.2f}")


# ---------------------------------------------------------------------------
# noise generators (for demonstrating the filters)
# ---------------------------------------------------------------------------

def add_salt_and_pepper(img, amount_pct: float = 2.0, seed: int = 0):
    """Set a random subset of pixels to 0 (pepper) or 255 (salt)."""
    a = np.array(img, dtype=np.uint8, copy=True)
    H, W = a.shape[:2]
    n = int(round(H * W * float(np.clip(amount_pct, 0, 100)) / 100.0))
    if n <= 0:
        return a
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(H * W, size=min(n, H * W), replace=False)
    ys, xs = np.unravel_index(idx, (H, W))
    half = n // 2
    a[ys[:half], xs[:half]] = 0
    a[ys[half:], xs[half:]] = 255
    return a


def add_gaussian_noise(img, sigma: float = 15.0, seed: int = 0):
    """g = f + eta,  eta ~ N(0, sigma^2)."""
    a = as_float(img)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, max(float(sigma), 0.0), a.shape)
    return as_uint8(a + noise)


# ---------------------------------------------------------------------------
# first derivative: Sobel / Scharr
# ---------------------------------------------------------------------------

def sobel_gradients(img, variant: str = "sobel", border: str = "reflect"):
    """Return (Gx, Gy, magnitude, angle) computed on the grayscale image."""
    g = to_gray(img)
    if variant == "scharr":
        kx, ky = SCHARR_X, SCHARR_Y
    else:
        kx, ky = SOBEL_X, SOBEL_Y
    gx = correlate(g, kx, border)
    gy = correlate(g, ky, border)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return gx, gy, mag, ang


def sobel_magnitude(img, threshold: float = 0.0,
                    variant: str = "sobel", border: str = "reflect"):
    """Edge strength, normalised to 0..255 and optionally thresholded."""
    _, _, mag, _ = sobel_gradients(img, variant, border)
    peak = float(mag.max())
    if peak <= 0:
        return np.zeros(mag.shape, dtype=np.uint8)
    m8 = mag / peak * 255.0
    if threshold > 0:
        m8 = np.where(m8 >= threshold, m8, 0.0)
    return as_uint8(m8)


def sobel_orientation(img, variant: str = "sobel",
                      border: str = "reflect") -> np.ndarray:
    """Direction of the gradient encoded as hue, strength as brightness."""
    _, _, mag, ang = sobel_gradients(img, variant, border)
    peak = float(mag.max())
    strength = mag / peak if peak > 0 else np.zeros_like(mag)

    # cyclic RGB from the angle (-pi..pi); 120 deg apart
    r = 0.5 + 0.5 * np.cos(ang)
    g = 0.5 + 0.5 * np.cos(ang - 2 * np.pi / 3)
    b = 0.5 + 0.5 * np.cos(ang + 2 * np.pi / 3)
    rgb = np.stack([r, g, b], axis=2)
    return as_uint8(rgb * strength[..., None] * 255.0)


# ---------------------------------------------------------------------------
# second derivative: Laplacian
# ---------------------------------------------------------------------------

def laplacian_response(img, kind: str = "4-neighbor",
                       border: str = "reflect"):
    """Signed second-derivative response (float, may be negative)."""
    if kind not in LAPLACIANS:
        raise ValueError(f"unknown Laplacian mask: {kind}")
    return correlate(img, LAPLACIANS[kind], border)


def laplacian_display(img, kind: str = "4-neighbor",
                      border: str = "reflect") -> np.ndarray:
    """Absolute response of the luminance channel, rescaled to 0..255.

    The notes define the second derivative on one intensity plane, so colour
    images are reduced to luminance here (sharpening itself still works per
    channel to preserve colour).
    """
    lap = correlate(to_gray(img), LAPLACIANS[kind], border)
    peak = float(np.abs(lap).max())
    if peak <= 0:
        return np.zeros(lap.shape[:2], dtype=np.uint8)
    return as_uint8(np.abs(lap) / peak * 255.0)


def laplacian_sharpen(img, kind: str = "4-neighbor", strength: float = 1.0,
                      border: str = "reflect") -> np.ndarray:
    """g = f - c * Laplacian(f)  for negative-centre masks.

    If a mask had a positive centre the sign flips, exactly as in the notes.
    """
    k = LAPLACIANS[kind]
    lap = correlate(img, k, border)
    sign = -1.0 if k[k.shape[0] // 2, k.shape[1] // 2] < 0 else 1.0
    return as_uint8(as_float(img) + sign * float(strength) * lap)
