"""Shared derivative kernels and helpers for the sharpening algorithms."""
from __future__ import annotations

import numpy as np

from ..core.utils import as_uint8, correlate, to_gray

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


def gradients(img, variant: str = "Sobel", border: str = "reflect"):
    """Return (Gx, Gy, magnitude, angle) on the luminance plane."""
    g = to_gray(img)
    if str(variant).lower().startswith("scharr"):
        kx, ky = SCHARR_X, SCHARR_Y
    else:
        kx, ky = SOBEL_X, SOBEL_Y
    gx = correlate(g, kx, border)
    gy = correlate(g, ky, border)
    return gx, gy, np.hypot(gx, gy), np.arctan2(gy, gx)


def orientation_rgb(angle: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Cyclic RGB hue for the angle, brightness for the magnitude."""
    peak = float(magnitude.max())
    strength = magnitude / peak if peak > 0 else np.zeros_like(magnitude)
    r = 0.5 + 0.5 * np.cos(angle)
    g = 0.5 + 0.5 * np.cos(angle - 2.0 * np.pi / 3.0)
    b = 0.5 + 0.5 * np.cos(angle + 2.0 * np.pi / 3.0)
    rgb = np.stack([r, g, b], axis=2)
    return as_uint8(rgb * strength[..., None] * 255.0)


def laplacian_abs(img, kind: str, border: str = "reflect") -> np.ndarray:
    """Absolute Laplacian of the luminance plane, scaled to 0..255."""
    lap = correlate(to_gray(img), LAPLACIANS[kind], border)
    peak = float(np.abs(lap).max())
    if peak <= 0:
        return np.zeros(lap.shape[:2], dtype=np.uint8)
    return as_uint8(np.abs(lap) / peak * 255.0)
