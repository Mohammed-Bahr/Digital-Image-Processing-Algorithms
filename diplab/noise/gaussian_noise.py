"""Additive Gaussian noise: g = f + eta, eta ~ N(0, sigma^2)."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import as_float, as_uint8, result


def add_gaussian(img, sigma: float, seed: int = 0) -> np.ndarray:
    a = as_float(img)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, max(0.0, float(sigma)), a.shape)
    return as_uint8(a + noise)


def run(img, params):
    sigma = float(params.get("sigma", 15.0))
    seed = int(params.get("seed", 0))
    out = add_gaussian(img, sigma, seed)
    log = (
        f"g = f + η,   η ~ N(0, σ²),   σ = {sigma:g} (seed {seed})\n"
        "Gaussian noise is added to every pixel independently.  A linear\n"
        "average reduces it, but also blurs edges; the median is better for\n"
        "impulse noise and the adaptive filter for mixed cases."
    )
    return result(out, log)


OPERATION = Operation(
    key="gauss_noise",
    name="Add Gaussian noise",
    category="Noise",
    description="g = f + η with η ~ N(0, σ²). Larger σ means a noisier image.",
    func=run,
    params=(
        Param("sigma", "σ", "float", 15, 0, 100, 1, 1),
        Param("seed", "Seed", "int", 0, 0, 9999, 1),
    ),
)
