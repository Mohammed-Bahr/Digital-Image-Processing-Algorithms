"""Fourier transforms and frequency-domain helpers.

The real filters use NumPy's FFT (fast, reliable).  The naive DFT
implementation kept for the lecture's worked example lives in
``dft_reference.py``.
"""
from __future__ import annotations

import numpy as np

from .utils import (EPS, as_float, as_uint8, next_power_of_two,
                    normalize_to_uint8, per_channel, to_gray)


# --------------------------------------------------------------------------
# transform wrappers
# --------------------------------------------------------------------------
def fft(x, axis: int = -1):
    return np.fft.fft(x, axis=axis)


def ifft(x, axis: int = -1):
    return np.fft.ifft(x, axis=axis)


def fft2(img):
    return np.fft.fft2(img)


def ifft2(F):
    return np.fft.ifft2(F)


def fftshift2(a):
    return np.fft.fftshift(a, axes=(-2, -1))


def ifftshift2(a):
    return np.fft.ifftshift(a, axes=(-2, -1))


# --------------------------------------------------------------------------
# padding and masks
# --------------------------------------------------------------------------
def padded_size(h: int, w: int, factor: float = 1.0) -> tuple[int, int]:
    """Next power-of-two size, optionally enlarged by ``factor``."""
    factor = max(1.0, float(factor))
    p = next_power_of_two(int(np.ceil(h * factor)))
    q = next_power_of_two(int(np.ceil(w * factor)))
    return max(p, 8), max(q, 8)


def pad_plane(plane: np.ndarray, p: int, q: int) -> np.ndarray:
    out = np.zeros((p, q), dtype=np.float64)
    out[:plane.shape[0], :plane.shape[1]] = plane
    return out


def distance_grid(p: int, q: int) -> np.ndarray:
    """Distance of every DFT sample from the (shifted) centre."""
    u = np.arange(p) - p // 2
    v = np.arange(q) - q // 2
    return np.hypot(u[:, None], v[None, :])


def mask_uint8(mask: np.ndarray) -> np.ndarray:
    return as_uint8(np.clip(mask, 0.0, 1.0) * 255.0)


def spectrum_uint8(F_centred: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    mag = np.log1p(np.abs(F_centred))
    peak = float(mag.max())
    if peak <= 0:
        return np.zeros(mag.shape, dtype=np.uint8)
    return as_uint8((mag / peak) ** max(gamma, EPS) * 255.0)


def spectrum_of(img: np.ndarray, pad_factor: float = 1.0):
    """Return (display spectrum, log-magnitude, padding)."""
    gray = to_gray(img)
    h, w = gray.shape
    p, q = padded_size(h, w, pad_factor)
    F = fftshift2(fft2(pad_plane(gray, p, q)))
    mag = np.log1p(np.abs(F))
    return normalize_to_uint8(mag), mag, (h, w, p, q)


# --------------------------------------------------------------------------
# generic filtering
# --------------------------------------------------------------------------
def apply_filter(img: np.ndarray, mask_fn, pad_factor: float = 1.0):
    """Multiply the centred spectrum by ``mask_fn((P, Q))``.

    Returns ``(output, mask, input_spectrum, output_spectrum)``.
    """
    a = as_float(img)
    h, w = a.shape[:2]
    p, q = padded_size(h, w, pad_factor)
    mask = np.asarray(mask_fn((p, q)), dtype=np.float64)
    if mask.shape != (p, q):
        raise ValueError("mask function returned the wrong shape")

    def filter_one(channel: np.ndarray) -> np.ndarray:
        F = fft2(pad_plane(channel, p, q))
        Fc = fftshift2(F)
        G = ifft2(ifftshift2(Fc * mask))
        return np.real(G)[:h, :w]

    out = per_channel(a, filter_one)

    gray = to_gray(img)
    F_in = fftshift2(fft2(pad_plane(gray, p, q)))
    spec_in = spectrum_uint8(F_in)
    spec_out = spectrum_uint8(F_in * mask)
    return as_uint8(out), mask, spec_in, spec_out


def filter_aux(mask: np.ndarray, spec_in: np.ndarray,
               spec_out: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Input |F|": spec_in,
        "Filter H": mask_uint8(mask),
        "Output |G|": spec_out,
    }


def kept_percent(mask: np.ndarray) -> float:
    return float(np.mean(mask)) * 100.0
