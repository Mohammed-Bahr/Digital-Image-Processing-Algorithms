"""Notch rejection: zero two symmetric spikes in the spectrum."""
from __future__ import annotations

from ..core.types import Operation, Param
from .common import PAD, notch_reject_mask, run_filter


def run(img, params):
    du = float(params.get("du", 40.0))
    dv = float(params.get("dv", 40.0))
    radius = float(params.get("radius", 6.0))
    pad = params.get("padding", "1")

    mask_fn = lambda shape: notch_reject_mask(shape, du, dv, radius)
    log = (
        f"Notch reject around offset (du, dv) = ({du:g}, {dv:g}) and its\n"
        f"mirror, radius = {radius:g}.\n"
        "Periodic noise appears as isolated bright spikes in the spectrum;\n"
        "zeroing both symmetric spikes removes the pattern without blurring\n"
        "the rest of the image."
    )
    return run_filter(img, mask_fn, pad, log)


OPERATION = Operation(
    key="notch",
    name="Notch reject (spikes)",
    category="Frequency Domain",
    description=(
        "Zero two symmetric spikes in the spectrum. Periodic noise "
        "produces exactly such isolated spikes."
    ),
    func=run,
    params=(
        Param("du", "Offset du", "float", 40, -400, 400, 1, 0),
        Param("dv", "Offset dv", "float", 40, -400, 400, 1, 0),
        Param("radius", "Radius", "float", 6, 1, 80, 1, 0),
        PAD,
    ),
)
