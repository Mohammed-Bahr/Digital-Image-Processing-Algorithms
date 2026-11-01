"""Gradient orientation shown as hue (Sobel or Scharr)."""
from __future__ import annotations

from ..core.types import Operation, Param
from ..core.utils import normalize_to_uint8, result
from .common import gradients, orientation_rgb


def run(img, params):
    variant = str(params.get("variant", "Sobel"))
    _, _, mag, angle = gradients(img, variant)
    out = orientation_rgb(angle, mag)
    log = (
        f"{variant} gradient direction: θ = atan2(Gy, Gx)\n"
        "The hue encodes θ over -π..π; brightness encodes |G|.\n"
        "An edge's gradient points across the edge, so the colour tells\n"
        "you the edge orientation (colours 180° apart are the same edge)."
    )
    return result(out, log,
                  aux={"Gradient magnitude": normalize_to_uint8(mag)})


OPERATION = Operation(
    key="sobel_dir",
    name="Sobel orientation",
    category="Sharpening",
    description=(
        "θ = atan2(Gy, Gx). Hue shows the gradient direction, "
        "brightness shows its magnitude."
    ),
    func=run,
    params=(
        Param("variant", "Operator", "choice", "Sobel",
              choices=("Sobel", "Scharr")),
    ),
)
