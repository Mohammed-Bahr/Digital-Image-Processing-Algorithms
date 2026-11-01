"""Extract one colour plane from an RGB image."""
from __future__ import annotations

from ..core.types import Operation, Param
from ..core.utils import as_float, result


_CHANNELS = {"Red": 0, "Green": 1, "Blue": 2}


def run(img, params):
    name = str(params.get("channel", "Red"))
    idx = _CHANNELS.get(name, 0)
    a = as_float(img)
    if a.ndim == 2:
        plane = a
        note = "Input is already grayscale; the plane is returned unchanged."
    else:
        plane = a[:, :, idx]
        note = f"Kept channel {idx} ({name})."
    log = (
        "A colour image is three stacked intensity matrices.\n"
        f"{note}\n"
        "Each plane can be enhanced or filtered on its own."
    )
    return result(plane, log)


OPERATION = Operation(
    key="channel",
    name="Extract colour channel",
    category="Foundations",
    description=(
        "Split the image into one of its three intensity planes. "
        "A colour image is three stacked matrices."
    ),
    func=run,
    params=(Param("channel", "Channel", "choice", "Red",
                  choices=("Red", "Green", "Blue")),),
)
