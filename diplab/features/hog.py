"""Histogram of Oriented Gradients (HOG).

The descriptor is computed from scratch, matching the notes:
unsigned gradients, soft orientation binning, 2x2-cell blocks and
L2 block normalisation.  OpenCV is used only for the convolution and
resizing helpers.
"""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import (as_float, correlate, normalize_to_uint8, resize,
                          result, to_gray)

_GRAD_X = np.array([[-1, 0, 1]], dtype=np.float64)
_GRAD_Y = np.array([[-1], [0], [1]], dtype=np.float64)


def _gradients(plane: np.ndarray, border: str = "reflect"):
    gx = correlate(plane, _GRAD_X, border)
    gy = correlate(plane, _GRAD_Y, border)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)      # unsigned 0..pi
    return mag, ang


def compute(img, cell: int = 8, bins: int = 9, block: int = 2,
            stride: int = 1, resize_to=None, border: str = "reflect") -> dict:
    """Compute the HOG vector and the per-cell histograms."""
    a = as_float(img)
    if a.ndim == 3:
        a = to_gray(a)
    if resize_to is not None:
        a = as_float(resize(a, int(resize_to[0]), int(resize_to[1]), "linear"))

    mag, ang = _gradients(a, border)
    h, w = mag.shape
    ny, nx = h // int(cell), w // int(cell)
    if ny < block or nx < block:
        raise ValueError("image is too small for the selected cell/block size")

    # Soft binning: each pixel splits its magnitude between two bins.
    pos = ang * (bins / np.pi)
    lower = np.floor(pos).astype(np.intp) % bins
    frac = pos - np.floor(pos)
    upper = (lower + 1) % bins

    hist = np.zeros((ny, nx, bins), dtype=np.float64)
    for j in range(ny):
        y0, y1 = j * cell, (j + 1) * cell
        for i in range(nx):
            x0, x1 = i * cell, (i + 1) * cell
            m = mag[y0:y1, x0:x1].ravel()
            lo = lower[y0:y1, x0:x1].ravel()
            hi = upper[y0:y1, x0:x1].ravel()
            f = frac[y0:y1, x0:x1].ravel()
            hist[j, i] = (
                np.bincount(lo, weights=m * (1.0 - f), minlength=bins)[:bins]
                + np.bincount(hi, weights=m * f, minlength=bins)[:bins]
            )

    bc = int(block)
    st = int(stride)
    by = (ny - bc) // st + 1
    bx = (nx - bc) // st + 1
    parts = []
    for j in range(by):
        for i in range(bx):
            part = hist[j * st:j * st + bc, i * st:i * st + bc].reshape(-1)
            norm = float(np.sqrt(np.dot(part, part) + 1e-6))
            parts.append(part / norm)
    vector = np.concatenate(parts) if parts else np.zeros(0)

    return {
        "vector": vector,
        "hist": hist,
        "cells": (ny, nx),
        "blocks": (by, bx),
        "cell_size": int(cell),
        "bins": int(bins),
        "block_cells": bc,
        "mag": mag,
        "angle": ang,
        "size": (h, w),
    }


def visualization(hist, bins: int = 9, scale: int = 10,
                  show_grid: bool = True) -> np.ndarray:
    """Draw one orientation fan per cell (the classic HOG picture)."""
    h = as_float(hist)
    ny, nx = h.shape[:2]
    out = np.full((ny * scale, nx * scale), 22, dtype=np.uint8)
    if h.max() <= 0:
        return out

    height, width = out.shape
    for j in range(ny):
        cy = j * scale + scale / 2.0
        for i in range(nx):
            cx = i * scale + scale / 2.0
            cell_hist = h[j, i]
            cmax = float(cell_hist.max())
            if cmax <= 0:
                continue
            for b in range(bins):
                v = cell_hist[b]
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
                out[np.clip(ys + 1, 0, height - 1), xs] = np.maximum(
                    out[np.clip(ys + 1, 0, height - 1), xs], val)
                out[ys, np.clip(xs + 1, 0, width - 1)] = np.maximum(
                    out[ys, np.clip(xs + 1, 0, width - 1)], val)
            out[int(cy), int(cx)] = 255

    if show_grid:
        out[::scale, :] = np.maximum(out[::scale, :], 42)
        out[:, ::scale] = np.maximum(out[:, ::scale], 42)
    return out


def run(img, params):
    cell = int(params.get("cell", 8))
    bins = int(params.get("bins", 9))
    block = int(params.get("block", 2))
    stride = int(params.get("stride", 1))
    resize_to = (64, 128) if str(params.get("resize", "64 x 128")).startswith("64") else None

    d = compute(img, cell, bins, block, stride, resize_to)
    scale = int(np.clip(440 // max(d["cells"][1], 1), 6, 26))
    viz = visualization(d["hist"], bins, scale)

    ny, nx = d["cells"]
    by, bx = d["blocks"]
    length = by * bx * (block * block) * bins
    mag = d["mag"]
    mag_img = normalize_to_uint8(mag, 0.0, max(float(mag.max()), 1e-12))

    log = (
        f"window {d['size'][1]}x{d['size'][0]} px, cell {cell}x{cell}, "
        f"{bins} bins, block {block}x{block} cells, stride {stride}.\n"
        f"cells: {nx} x {ny} (={nx * ny}), "
        f"block positions: {bx} x {by} (={bx * by}).\n"
        f"descriptor length = {bx}·{by}·({block}²·{bins}) = {length}.\n"
        "9 bins = 20° each; each pixel votes softly for the two nearest\n"
        "bins; each 2x2 block is L2-normalised to cancel contrast changes.\n"
        "The picture shows one orientation fan per cell: longer lines mean\n"
        "stronger gradient votes in that direction."
    )
    return result(viz, log,
                  aux={"Gradient magnitude": mag_img},
                  meta={"vector": d["vector"], "hist": d["hist"],
                        "cells": d["cells"], "blocks": d["blocks"],
                        "length": int(length)})


OPERATION = Operation(
    key="hog",
    name="HOG descriptor",
    category="Features",
    description=(
        "Histogram of Oriented Gradients: gradients -> 8x8 cell "
        "histograms -> overlapping 2x2 block normalisation."
    ),
    func=run,
    params=(
        Param("resize", "Resize first", "choice", "64 x 128",
              choices=("64 x 128", "keep size")),
        Param("cell", "Cell size", "int", 8, 4, 32, 4),
        Param("bins", "Orientation bins", "int", 9, 4, 18, 1),
        Param("block", "Block (cells)", "int", 2, 1, 4, 1),
        Param("stride", "Block stride", "int", 1, 1, 3, 1),
    ),
)
