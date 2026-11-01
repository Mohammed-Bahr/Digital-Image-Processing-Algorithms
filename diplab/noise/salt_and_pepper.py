"""Salt-and-pepper impulse noise."""
from __future__ import annotations

import numpy as np

from ..core.types import Operation, Param
from ..core.utils import as_uint8, result


def add_salt_pepper(img, amount: float, seed: int = 0) -> np.ndarray:
    a = as_uint8(img)
    rng = np.random.default_rng(int(seed))
    total = a.size if a.ndim == 2 else a.shape[0] * a.shape[1] * a.shape[2]
    n = int(round(total * max(0.0, min(float(amount), 100.0)) / 100.0))
    if n <= 0:
        return a.copy()
    flat = a.reshape(-1)
    idx = rng.choice(flat.size, size=min(n, flat.size), replace=False)
    half = idx.size // 2
    flat[idx[:half]] = 0
    flat[idx[half:]] = 255
    return flat.reshape(a.shape)


def run(img, params):
    amount = float(params.get("amount", 3.0))
    seed = int(params.get("seed", 0))
    out = add_salt_pepper(img, amount, seed)
    log = (
        f"Salt & pepper noise, {amount:g}% of the pixels (seed {seed}).\n"
        "Half of the selected pixels become 0 (pepper), half become 255\n"
        "(salt).  Impulse noise is best removed by an order filter such as\n"
        "the median, not by a linear average."
    )
    return result(out, log)


OPERATION = Operation(
    key="salt",
    name="Add salt & pepper noise",
    category="Noise",
    description="Random pixels are forced to 0 (pepper) or 255 (salt).",
    func=run,
    params=(
        Param("amount", "Noise %", "float", 3, 0, 30, 0.5, 1),
        Param("seed", "Seed", "int", 0, 0, 9999, 1),
    ),
)
