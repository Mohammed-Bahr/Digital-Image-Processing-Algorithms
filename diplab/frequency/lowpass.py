"""Low-pass filtering: keep the centre of the spectrum."""
from __future__ import annotations

from ..core.types import Operation, Param
from .common import KIND, PAD, lowpass_mask, run_filter


def run(img, params):
    kind = str(params.get("kind", "butterworth"))
    d0 = float(params.get("d0", 30.0))
    order = float(params.get("order", 2.0))
    pad = params.get("padding", "1")

    mask_fn = lambda shape: lowpass_mask(shape, d0, order, kind)
    if kind == "ideal":
        formula = "H = 1 if D(u,v) ≤ D0 else 0"
    elif kind == "butterworth":
        formula = f"H = 1 / (1 + [D/D0]^(2n)),   n = {order:g}"
    else:
        formula = "H = exp(-D² / 2D0²)"
    log = (
        f"{kind} low-pass (smoothing / blurring)\n"
        f"{formula}\n"
        f"D0 = {d0:g}, padding factor = {pad}\n"
        "Pipeline: f -> DFT -> multiply by H -> inverse DFT -> g.\n"
        "The centre (low frequencies) survives; the outside (edges, detail,\n"
        "noise) is attenuated.  Ideal filters ring, Gaussian filters do not."
    )
    return run_filter(img, mask_fn, pad, log)


OPERATION = Operation(
    key="lowpass",
    name="Low-pass filter",
    category="Frequency Domain",
    description=(
        "Keep the centre, remove the outside: smooths/blurs the image. "
        "Ideal rings, Butterworth rolls off, Gaussian is smoothest."
    ),
    func=run,
    params=(
        KIND,
        Param("d0", "Cutoff D0", "float", 30, 1, 400, 1, 0),
        Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
        PAD,
    ),
)
