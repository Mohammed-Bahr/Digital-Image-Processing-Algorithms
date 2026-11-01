"""High-pass filtering: remove the centre of the spectrum."""
from __future__ import annotations

from ..core.types import Operation, Param
from ..core.utils import normalize_to_uint8
from .common import KIND, PAD, highpass_mask, run_filter


def run(img, params):
    kind = str(params.get("kind", "butterworth"))
    d0 = float(params.get("d0", 30.0))
    order = float(params.get("order", 2.0))
    pad = params.get("padding", "1")
    rescale = bool(params.get("rescale", True))

    mask_fn = lambda shape: highpass_mask(shape, d0, order, kind)
    if kind == "ideal":
        formula = "H = 0 if D(u,v) ≤ D0 else 1"
    elif kind == "butterworth":
        formula = f"H = 1 / (1 + [D0/D]^(2n)),   n = {order:g}"
    else:
        formula = "H = 1 - exp(-D² / 2D0²)"
    log = (
        f"{kind} high-pass (sharpening / edges)\n"
        f"{formula}\n"
        f"D0 = {d0:g}, padding factor = {pad}\n"
        "The DC term is zeroed, so smooth regions go black and only\n"
        "outlines remain.  The result is naturally dim; rescaling for\n"
        "display stretches it back to 0..255."
    )
    res = run_filter(img, mask_fn, pad, log)
    if rescale and res.image.max() > 0:
        res.image = normalize_to_uint8(res.image)
        res.log += ("\nDisplay: output linearly rescaled to the full 0..255 "
                    "range (high-pass results are naturally dim).")
    return res


OPERATION = Operation(
    key="highpass",
    name="High-pass filter",
    category="Frequency Domain",
    description=(
        "Remove the centre, keep edges. The DC term is zeroed, so smooth "
        "regions go black and only outlines remain."
    ),
    func=run,
    params=(
        KIND,
        Param("d0", "Cutoff D0", "float", 30, 1, 400, 1, 0),
        Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
        Param("rescale", "Rescale for display", "bool", True),
        PAD,
    ),
)
