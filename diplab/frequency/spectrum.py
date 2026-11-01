"""Display the centred log-magnitude Fourier spectrum."""
from __future__ import annotations

from ..core.types import Operation
from ..core.transforms import spectrum_of
from ..core.utils import result
from .common import PAD, pad_factor


def run(img, params):
    pad = pad_factor(params.get("padding", "1"))
    spec, mag, (h, w, p, q) = spectrum_of(img, pad)
    log = (
        "Centred Fourier spectrum, displayed as log(1 + |F|).\n"
        f"Image {w}x{h} zero-padded to {q}x{p}.\n"
        "Centre = DC component (average brightness).\n"
        "Near the centre = low frequencies (smooth regions).\n"
        "Far from the centre = high frequencies (edges, detail, noise).\n"
        "The transform is lossless: the inverse recovers the image."
    )
    return result(spec, log, meta={"spectrum": mag})


OPERATION = Operation(
    key="spectrum",
    name="Fourier spectrum",
    category="Frequency Domain",
    description=(
        "Centred log-magnitude spectrum. Centre = DC/average "
        "brightness, outside = edges, detail and noise."
    ),
    func=run,
    params=(PAD,),
)
