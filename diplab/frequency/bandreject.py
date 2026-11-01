"""Band-reject filtering: remove a ring of frequencies."""
from __future__ import annotations

from ..core.types import Operation, Param
from .common import KIND, PAD, band_mask, run_filter


def run(img, params):
    kind = str(params.get("kind", "butterworth"))
    d0 = float(params.get("d0", 15.0))
    d1 = float(params.get("d1", 55.0))
    order = float(params.get("order", 2.0))
    pad = params.get("padding", "1")

    mask_fn = lambda shape: band_mask(shape, d0, d1, order, kind, True)
    log = (
        f"{kind} band-reject: remove D0 ≤ D ≤ D1  ({d0:g} .. {d1:g})\n"
        f"order = {order:g}, padding factor = {pad}\n"
        "This is the complement of the band-pass mask.\n"
        "Periodic noise creates a ring of spikes in the spectrum; removing\n"
        "that ring is the classic band-reject application."
    )
    return run_filter(img, mask_fn, pad, log)


OPERATION = Operation(
    key="bandreject",
    name="Band-reject (notch band)",
    category="Frequency Domain",
    description=(
        "Remove a ring of frequencies instead of keeping it. Used for "
        "periodic noise."
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
