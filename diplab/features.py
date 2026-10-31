"""Local image features -- notes 03 (Lecture 5).

Two algorithms built from scratch:

* **Harris corner detector** -- structure tensor M, response
  R = det(M) - k * trace(M)^2, non-maximum suppression.
* **HOG descriptor** -- gradients, soft-binned cell histograms, overlapping
  block normalisation, visualisation.

No OpenCV is used.
"""

from __future__ import annotations

import numpy as np

from .spatial import sobel_gradients
from .utils import (as_float, as_uint8, correlate, draw_markers,
                    gaussian_kernel_2d, maximum_filter, resize_bilinear,
                    to_gray)


# ---------------------------------------------------------------------------
# Harris
# ---------------------------------------------------------------------------

def harris_response(img, k: float = 0.04, window: int = 3,
                    gaussian: bool = True) -> np.ndarray:
    """Compute the Harris response R for every pixel.

    M = sum_w [ Ix^2  IxIy ; IxIy  Iy^2 ]
    R = det(M) - k * trace(M)^2
    """
    g = to_gray(img)
    gx, gy, _, _ = sobel_gradients(g)

    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    if gaussian:
        w = gaussian_kernel_2d(window, sigma=max(window / 3.0, 0.8))
    else:
        w = np.ones((window, window), dtype=np.float64)

    a = correlate(ixx, w)
    b = correlate(ixy, w)
    c = correlate(iyy, w)

    det_m = a * c - b * b
    trace_m = a + c
    return det_m - float(k) * trace_m * trace_m


def harris_corners(img, k: float = 0.04, window: int = 3,
                   threshold_pct: float = 99.0, min_distance: int = 5,
                   max_points: int = 400):
    """Return (list of (row, col) corners, response map)."""
    r = harris_response(img, k=k, window=window)
    if not np.isfinite(r).any():
        return [], r

    local_max = maximum_filter(r, size=3)
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


def harris_visualization(img, k: float = 0.04, window: int = 3,
                         threshold_pct: float = 99.0,
                         min_distance: int = 5, radius: int = 5):
    """Draw detected corners on top of the image; also return the points."""
    points, r = harris_corners(img, k, window, threshold_pct, min_distance)
    return draw_markers(img, points, radius=radius), points, r


def harris_response_image(img, k: float = 0.04, window: int = 3) -> np.ndarray:
    """Positive part of R rescaled to 0..255 for display."""
    r = harris_response(img, k=k, window=window)
    r = np.where(np.isfinite(r), r, 0.0)
    peak = float(r.max())
    if peak <= 0:
        return np.zeros(r.shape, dtype=np.uint8)
    return as_uint8(np.clip(r, 0, peak) / peak * 255.0)


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------

def _gradients(plane, border: str = "reflect"):
    """Ix, Iy with the lecture's 1-D kernels [-1, 0, 1]."""
    kx = np.array([[-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1], [0], [1]], dtype=np.float64)
    gx = correlate(plane, kx, border)
    gy = correlate(plane, ky, border)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)     # unsigned 0..180 degrees
    return mag, ang


def hog(img, cell: int = 8, bins: int = 9, block_cells: int = 2,
        stride: int = 1, resize=None, border: str = "reflect") -> dict:
    """Compute a HOG descriptor.

    Parameters
    ----------
    cell        : cell edge in pixels (8 in the notes)
    bins        : orientation bins (9 in the notes -> 20 deg each)
    block_cells : cells per block edge (2 -> 2x2 cells = 16x16 px)
    stride      : block stride in cells (1 -> overlapping blocks)
    resize      : optional (width, height) to resize first; (64, 128) matches
                  the lecture's fixed 1:2 window.

    Returns a dict with the vector, cell histograms and diagnostics.
    """
    a = as_float(img)
    if resize is not None:
        a = resize_bilinear(a, int(resize[0]), int(resize[1]))

    color_note = ""
    if a.ndim == 3:
        mags, angs = [], []
        for c in range(a.shape[2]):
            m, t = _gradients(a[..., c], border)
            mags.append(m)
            angs.append(t)
        m_all = np.stack(mags, axis=-1)
        a_all = np.stack(angs, axis=-1)
        idx = np.argmax(m_all, axis=-1)
        mag = np.take_along_axis(m_all, idx[..., None], axis=2)[..., 0]
        ang = np.take_along_axis(a_all, idx[..., None], axis=2)[..., 0]
        color_note = "colour: per-pixel max-gradient channel (lecture variant)"
    else:
        mag, ang = _gradients(a, border)

    H, W = mag.shape
    Ny, Nx = H // int(cell), W // int(cell)
    if Ny < block_cells or Nx < block_cells:
        raise ValueError("image is too small for the selected cell/block size")

    # soft binning: each pixel splits its magnitude between two bins
    pos = ang * (bins / np.pi)
    lower = np.floor(pos).astype(np.intp) % bins
    frac = pos - np.floor(pos)
    upper = (lower + 1) % bins

    hist = np.zeros((Ny, Nx, bins), dtype=np.float64)
    for j in range(Ny):
        y0, y1 = j * cell, (j + 1) * cell
        for i in range(Nx):
            x0, x1 = i * cell, (i + 1) * cell
            m = mag[y0:y1, x0:x1].ravel()
            lo = lower[y0:y1, x0:x1].ravel()
            hi = upper[y0:y1, x0:x1].ravel()
            f = frac[y0:y1, x0:x1].ravel()
            hist[j, i] = (np.bincount(lo, weights=m * (1 - f), minlength=bins)[:bins] +
                          np.bincount(hi, weights=m * f, minlength=bins)[:bins])

    # overlapping blocks, L2 normalisation
    bc = int(block_cells)
    st = int(stride)
    by = (Ny - bc) // st + 1
    bx = (Nx - bc) // st + 1
    parts = []
    for j in range(by):
        for i in range(bx):
            block = hist[j * st:j * st + bc, i * st:i * st + bc].reshape(-1)
            norm = float(np.sqrt(np.dot(block, block) + 1e-6))
            parts.append(block / norm)
    vector = np.concatenate(parts) if parts else np.zeros(0)

    return {
        "vector": vector,
        "hist": hist,
        "cells": (Ny, Nx),
        "blocks": (by, bx),
        "cell_size": int(cell),
        "bins": int(bins),
        "block_cells": bc,
        "mag": mag,
        "angle": ang,
        "color_note": color_note,
        "size": (H, W),
    }


def hog_visualization(hist, bins: int = 9, scale: int = 10,
                      show_grid: bool = True) -> np.ndarray:
    """Draw the classic HOG "star" picture: one orientation fan per cell.

    Each cell is normalised by its own strongest bin so faint cells stay
    readable, and lines are two pixels wide.
    """
    h = as_float(hist)
    Ny, Nx = h.shape[:2]
    out = np.full((Ny * scale, Nx * scale), 22, dtype=np.uint8)
    if h.max() <= 0:
        return out

    H, W = out.shape
    for j in range(Ny):
        cy = j * scale + scale / 2.0
        for i in range(Nx):
            cx = i * scale + scale / 2.0
            cell = h[j, i]
            cmax = float(cell.max())
            if cmax <= 0:
                continue
            for b in range(bins):
                v = cell[b]
                if v <= 0:
                    continue
                ratio = min(float(v) / cmax, 1.0)
                angle = (b + 0.5) * np.pi / bins
                length = ratio * (scale * 0.5 - 1.0)
                dy = np.sin(angle) * length
                dx = np.cos(angle) * length

                n = max(int(2 * length) + 3, 3)
                ts = np.linspace(-1.0, 1.0, n)
                ys = np.rint(cy + ts * dy).astype(np.intp)
                xs = np.rint(cx + ts * dx).astype(np.intp)
                ys = np.clip(ys, j * scale, (j + 1) * scale - 1)
                xs = np.clip(xs, i * scale, (i + 1) * scale - 1)
                val = np.uint8(np.clip(90 + 165.0 * ratio, 0, 255))
                out[ys, xs] = np.maximum(out[ys, xs], val)
                # thicken the stroke by one pixel
                ys2 = np.clip(ys + 1, 0, H - 1)
                out[ys2, xs] = np.maximum(out[ys2, xs], val)
                xs2 = np.clip(xs + 1, 0, W - 1)
                out[ys, xs2] = np.maximum(out[ys, xs2], val)

            # centre dot so the cell layout is visible
            out[int(cy), int(cx)] = 255

    if show_grid:
        out[::scale, :] = np.maximum(out[::scale, :], 42)
        out[:, ::scale] = np.maximum(out[:, ::scale], 42)

    return out
