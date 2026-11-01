"""Naive DFT/IDFT kept for the lecture's worked example.

The filters themselves use NumPy's FFT; this module shows the definition
F(u) = sum_x f(x) exp(-j 2 pi u x / N) explicitly.  It is O(N^2) and is
only used for the small example in the GUI's Tools dialog.
"""
from __future__ import annotations

import numpy as np


def dft_matrix(n: int) -> np.ndarray:
    x = np.arange(n)
    u = x.reshape(-1, 1)
    return np.exp(-2j * np.pi * u * x / n)


def dft_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.complex128)
    return dft_matrix(a.size) @ a


def idft_1d(X) -> np.ndarray:
    a = np.asarray(X, dtype=np.complex128)
    return (np.conj(dft_matrix(a.size)) @ a) / a.size


def dft_2d(img) -> np.ndarray:
    a = np.asarray(img, dtype=np.complex128)
    m, n = a.shape
    return dft_matrix(m) @ a @ dft_matrix(n).T


def idft_2d(F) -> np.ndarray:
    a = np.asarray(F, dtype=np.complex128)
    m, n = a.shape
    return (np.conj(dft_matrix(m)) @ a @ np.conj(dft_matrix(n)).T) / (m * n)


def dft_worked_example(f=(1, 2, 4, 4)) -> str:
    """The 4-point example from the notes, formatted for the GUI."""
    x = np.asarray(f, dtype=np.complex128)
    X = dft_1d(x)
    back = idft_1d(X)

    def fmt(values):
        parts = []
        for v in values:
            re, im = float(v.real), float(v.imag)
            if abs(re) < 1e-9:
                re = 0.0
            if abs(im) < 1e-9:
                im = 0.0
            if abs(im) < 1e-9:
                parts.append(f"{re:g}")
            else:
                sign = "+" if im >= 0 else "-"
                parts.append(f"{re:g} {sign} {abs(im):g}j")
        return "[" + ", ".join(parts) + "]"

    lines = [
        "Worked 1-D DFT example from the notes",
        "",
        "f(n) = " + fmt(x),
        "N    = 4",
        "",
        "F(u) = sum_{n=0}^{N-1} f(n) exp(-j 2 pi u n / N)",
        "",
        "F = " + fmt(X),
        "",
        "Expected from the notes: [11, -3 + 2j, -1, -3 - 2j]",
        "",
        "Inverse check: idft(F) = " + fmt(back),
        "",
        "The GUI's frequency filters use NumPy's FFT (same transform,",
        "much faster).  This naive O(N^2) version is only for the example.",
    ]
    return "\n".join(lines)
