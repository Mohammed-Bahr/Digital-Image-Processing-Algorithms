"""Frequency-domain processing -- notes 02 (Lecture 4).

Contains a from-scratch DFT (naive, for the worked example), a radix-2 FFT,
spectrum visualisation, low/high/band/notch masks, and homomorphic filtering.
"""

from __future__ import annotations

import numpy as np

from .utils import (as_float, as_uint8, next_power_of_two, normalize_to_uint8,
                    per_channel, to_gray)

EPS = 1e-12


# ---------------------------------------------------------------------------
# 1-D transforms
# ---------------------------------------------------------------------------

def dft_matrix(n: int, inverse: bool = False) -> np.ndarray:
    """N x N DFT matrix: W[u, x] = exp(-2*pi*j*u*x/N)."""
    k = np.arange(n)
    sign = 1.0 if inverse else -1.0
    return np.exp(sign * 2j * np.pi * np.outer(k, k) / n)


def dft_1d(x) -> np.ndarray:
    """Naive O(N^2) forward DFT -- used for the lecture's worked example."""
    a = np.asarray(x, dtype=np.complex128).ravel()
    return dft_matrix(len(a)) @ a


def idft_1d(X) -> np.ndarray:
    """Naive inverse DFT."""
    a = np.asarray(X, dtype=np.complex128).ravel()
    return (np.conj(dft_matrix(len(a))).T @ a) / len(a)


def dft_2d(img) -> np.ndarray:
    """Naive 2-D DFT:  F = Wm @ f @ Wn^T  (O(M^2 N + M N^2))."""
    f = as_float(img)
    if f.ndim != 2:
        raise ValueError("dft_2d expects a 2-D image")
    M, N = f.shape
    return dft_matrix(M) @ f @ dft_matrix(N).T


def idft_2d(F) -> np.ndarray:
    """Naive inverse 2-D DFT."""
    F = np.asarray(F, dtype=np.complex128)
    M, N = F.shape
    back = np.conj(dft_matrix(M)).T @ F @ np.conj(dft_matrix(N))
    return np.real(back / (M * N))


# ---------------------------------------------------------------------------
# radix-2 FFT (iterative, vectorised over all leading axes)
# ---------------------------------------------------------------------------

def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.uint64)
    out = np.zeros(n, dtype=np.uint64)
    for _ in range(bits):
        out = (out << 1) | (idx & 1)
        idx >>= 1
    return out.astype(np.intp)


def fft(x, axis: int = -1) -> np.ndarray:
    """Cooley-Tukey radix-2 FFT along *axis* (length must be a power of 2)."""
    a = np.asarray(x, dtype=np.complex128)
    n = a.shape[axis]
    if n == 0:
        return a.copy()
    if n & (n - 1):
        raise ValueError(f"FFT length must be a power of two (got {n})")

    a = np.swapaxes(a, axis, -1)
    shape = a.shape
    a = np.ascontiguousarray(a.reshape(-1, n))
    if n > 1:
        a = a[:, _bit_reverse_indices(n)]

    m = 2
    while m <= n:
        half = m // 2
        w = np.exp(-2j * np.pi * np.arange(half) / m)
        y = a.reshape(a.shape[0], -1, m)
        u = y[:, :, :half].copy()
        v = y[:, :, half:] * w
        y[:, :, :half] = u + v
        y[:, :, half:] = u - v
        m <<= 1

    a = a.reshape(shape)
    return np.swapaxes(a, -1, axis)


def ifft(x, axis: int = -1) -> np.ndarray:
    """Inverse FFT via the conjugate trick:  ifft(x) = conj(fft(conj(x))) / N."""
    a = np.asarray(x, dtype=np.complex128)
    n = a.shape[axis]
    return np.conj(fft(np.conj(a), axis=axis)) / n


def fft2(img) -> np.ndarray:
    """2-D FFT (rows first, then columns)."""
    a = np.asarray(img, dtype=np.complex128)
    return fft(fft(a, axis=-1), axis=0)


def ifft2(F) -> np.ndarray:
    """2-D inverse FFT."""
    a = np.asarray(F, dtype=np.complex128)
    return ifft(ifft(a, axis=-1), axis=0)


def fftshift2(arr) -> np.ndarray:
    """Move the DC term (at index 0,0) to the centre."""
    a = np.asarray(arr)
    return np.roll(a, (a.shape[-2] // 2, a.shape[-1] // 2), axis=(-2, -1))


def ifftshift2(arr) -> np.ndarray:
    """Inverse of :func:`fftshift2`."""
    a = np.asarray(arr)
    return np.roll(a, (-(a.shape[-2] // 2), -(a.shape[-1] // 2)), axis=(-2, -1))


# ---------------------------------------------------------------------------
# padding + spectra
# ---------------------------------------------------------------------------

def padded_size(h: int, w: int, factor: float = 1.0) -> tuple[int, int]:
    """Next power-of-two (P, Q) at least as large as the image (times factor)."""
    factor = max(1.0, float(factor))
    P = next_power_of_two(max(2, int(np.ceil(h * factor))))
    Q = next_power_of_two(max(2, int(np.ceil(w * factor))))
    return P, Q


def _pad_plane(plane, P: int, Q: int) -> np.ndarray:
    out = np.zeros((P, Q), dtype=np.float64)
    h, w = plane.shape
    out[:h, :w] = plane
    return out


def spectrum_of(img, pad_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (displayable centred log-spectrum, complex centred F)."""
    g = to_gray(img)
    H, W = g.shape
    P, Q = padded_size(H, W, pad_factor)
    F = fftshift2(fft2(_pad_plane(g, P, Q)))
    mag = np.log1p(np.abs(F))
    return normalize_to_uint8(mag, 0.0, max(float(mag.max()), EPS)), F


def spectrum_uint8(F) -> np.ndarray:
    """Log magnitude of an already-centred spectrum, scaled to 0..255."""
    mag = np.log1p(np.abs(F))
    return normalize_to_uint8(mag, 0.0, max(float(mag.max()), EPS))


def mask_uint8(mask) -> np.ndarray:
    return as_uint8(np.asarray(mask, dtype=np.float64) * 255.0)


# ---------------------------------------------------------------------------
# distance from the centre and filter masks
# ---------------------------------------------------------------------------

def distance_grid(shape) -> np.ndarray:
    """D(u, v) = sqrt((u - P/2)^2 + (v - Q/2)^2) after centring."""
    P, Q = shape
    u = np.arange(P)[:, None] - P / 2.0
    v = np.arange(Q)[None, :] - Q / 2.0
    return np.hypot(u, v)


def lowpass_mask(shape, d0: float, n: float = 2.0,
                 kind: str = "ideal") -> np.ndarray:
    """Ideal / Butterworth / Gaussian low-pass H(u, v)."""
    D = distance_grid(shape)
    d0 = max(float(d0), EPS)
    if kind == "ideal":
        return (D <= d0).astype(np.float64)
    if kind == "butterworth":
        return 1.0 / (1.0 + (D / d0) ** (2.0 * max(float(n), 0.01)))
    if kind == "gaussian":
        return np.exp(-(D * D) / (2.0 * d0 * d0))
    raise ValueError(f"unknown low-pass kind: {kind}")


def highpass_mask(shape, d0: float, n: float = 2.0,
                  kind: str = "ideal") -> np.ndarray:
    """Every high-pass is the complement of its low-pass partner."""
    return 1.0 - lowpass_mask(shape, d0, n, kind)


def band_mask(shape, d0: float, d1: float, n: float = 2.0,
              kind: str = "ideal", reject: bool = False) -> np.ndarray:
    """Band-pass = low-pass(D1) * high-pass(D0); ``reject`` complements it."""
    lo, hi = sorted((max(d0, EPS), max(d1, EPS)))
    bp = lowpass_mask(shape, hi, n, kind) * (1.0 - lowpass_mask(shape, lo, n, kind))
    return 1.0 - bp if reject else bp


def notch_reject_mask(shape, du: float, dv: float,
                      radius: float) -> np.ndarray:
    """Remove two symmetric discs around (P/2 + dv, Q/2 + du) and its mirror."""
    P, Q = shape
    yy = np.arange(P)[:, None] - P / 2.0
    xx = np.arange(Q)[None, :] - Q / 2.0
    d1 = np.hypot(yy - dv, xx - du)
    d2 = np.hypot(yy + dv, xx + du)
    r = max(float(radius), EPS)
    mask = np.ones(shape, dtype=np.float64)
    mask[d1 <= r] = 0.0
    mask[d2 <= r] = 0.0
    return mask


# ---------------------------------------------------------------------------
# the filtering pipeline:  f -> DFT -> xH -> IDFT -> g
# ---------------------------------------------------------------------------

def frequency_filter(img, mask_fn, pad_factor: float = 1.0):
    """Apply ``mask_fn(shape) -> H`` to the spectrum of *img*.

    Returns (filtered uint8 image, mask, input spectrum, filtered spectrum).
    """
    a = as_float(img)
    H, W = a.shape[:2]
    P, Q = padded_size(H, W, pad_factor)
    mask = np.asarray(mask_fn((P, Q)), dtype=np.float64)

    def one(ch):
        # FFT puts DC at (0,0); shift it to the centre so it lines up with H.
        F = fft2(_pad_plane(ch, P, Q))
        Fc = fftshift2(F)
        return np.real(ifft2(ifftshift2(Fc * mask))[:H, :W])

    out = per_channel(a, one)

    g = to_gray(img)
    F_in = fftshift2(fft2(_pad_plane(g, P, Q)))
    spec_in = spectrum_uint8(F_in)
    spec_out = spectrum_uint8(F_in * mask)
    return as_uint8(out), mask, spec_in, spec_out


def homomorphic(img, gamma_l: float = 0.5, gamma_h: float = 2.0,
                d0: float = 30.0, pad_factor: float = 1.0) -> np.ndarray:
    """Homomorphic filtering.

    log -> DFT -> high-pass-ish H -> IDFT -> exp.
    H(u,v) = (gamma_H - gamma_L) * (1 - exp(-D^2 / 2 D0^2)) + gamma_L
    """
    a = as_float(img)
    H, W = a.shape[:2]
    P, Q = padded_size(H, W, pad_factor)
    D = distance_grid((P, Q))
    d0 = max(float(d0), EPS)
    gl, gh = float(gamma_l), float(gamma_h)
    Hf = (gh - gl) * (1.0 - np.exp(-(D * D) / (2.0 * d0 * d0))) + gl

    def one(ch):
        z = np.log1p(np.maximum(ch, 0.0))
        F = fft2(_pad_plane(z, P, Q))
        s = np.real(ifft2(ifftshift2(fftshift2(F) * Hf))[:H, :W])
        s = np.clip(s, -20.0, 20.0)
        return np.expm1(s)

    out = per_channel(a, one)
    # Display rescale: homomorphic output is usually very compressed, so map
    # the 2nd..98th percentile to 0..255 (outliers would ruin a min/max map).
    lo, hi = np.percentile(out, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(out.min()), float(out.max())
    if hi <= lo:
        return as_uint8(out)
    return as_uint8(np.clip((out - lo) / (hi - lo) * 255.0, 0, 255))


# ---------------------------------------------------------------------------
# little demo used by the "Tools -> DFT worked example" dialog
# ---------------------------------------------------------------------------

def dft_worked_example() -> str:
    """Reproduce Example 4.1 from the notes: f = [1, 2, 4, 4]."""
    f = np.array([1, 2, 4, 4], dtype=np.float64)
    F = dft_1d(f)
    back = idft_1d(F)
    lines = [
        "Example 4.1 — forward DFT",
        f"f(x) = {f.astype(int).tolist()},  M = {len(f)}",
        "",
        "F(u) =",
    ]
    for u, val in enumerate(F):
        lines.append(f"  F({u}) = {val.real:+.4g} {val.imag:+.4g}j")
    lines += [
        "",
        "|F(u)| =",
        "  " + ", ".join(f"{abs(v):.4g}" for v in F),
        "",
        "Inverse DFT (should recover [1, 2, 4, 4]):",
        "  " + ", ".join(f"{v.real:.6g}" for v in back),
        "",
        "Conjugate symmetry check: F(3) == conj(F(1)): "
        f"{np.allclose(F[3], np.conj(F[1]))}",
        "",
        "2-D check on the note's 3x2 image (F(0,0) should be 10):",
    ]
    sample = np.array([[5, 10, 20], [5, 15, 5]], dtype=np.float64)
    F2 = dft_2d(sample) / sample.size       # note's 1/MN convention
    lines.append(f"  F(0,0) = {F2[0, 0].real:.4g}")
    return "\n".join(lines)
