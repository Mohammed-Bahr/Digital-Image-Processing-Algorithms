"""Band-pass filtering: keep a ring of frequencies."""
from __future__ import annotations

from ..core.types import Operation, Param
from .common import KIND, PAD, band_mask, run_filter


def run(img, params):
    kind = str(params.get("kind", "butterworth"))
    d0 = float(params.get("d0", 15.0))
    d1 = float(params.get("d1", 55.0))
    order = float(params.get("order", 2.0))
    pad = params.get("padding", "1")

    mask_fn = lambda shape: band_mask(shape, d0, d1, order, kind, False)
    log = (
        f"{kind} band-pass: keep D0 ≤ D ≤ D1  ({d0:g} .. {d1:g})\n"
        f"order = {order:g}, padding factor = {pad}\n"
        "Band-pass = low-pass(D1) · high-pass(D0).\n"
        "It suppresses low frequencies (overall brightness) and high\n"
        "frequencies (noise) at the same time.  Useful for texture and\n"
        "for isolating a periodic pattern."
    )
    return run_filter(img, mask_fn, pad, log)


OPERATION = Operation(
    key="bandpass",
    name="Band-pass filter",
    category="Frequency Domain",
    description=(
        "Keep only D0 ≤ D ≤ D1. Suppresses low frequencies (edges) and "
        "high frequencies (noise) at the same time."
    ),
    func=run,
    params=(
        KIND,
        Param("d0", "Lower D0", "float", 15, 1, 400, 1, 0),
        Param("d1", "Upper D1", "float", 55, 2, 500, 1, 0),
        Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
        PAD,
    ),
)
