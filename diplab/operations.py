"""Operation registry: every note algorithm exposed as a GUI action.

Each operation declares its parameters, a short theory description (taken from
the notes), and a function ``(image, params) -> OpResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import enhancement as en
from . import features as ft
from . import frequency as fr
from . import spatial as sp
from .utils import (EPS, as_float, as_uint8, describe_kernel, draw_markers,
                    normalize_to_uint8, to_gray)


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class Param:
    key: str
    label: str
    kind: str = "int"            # int | float | choice | bool
    default: Any = 0
    minimum: float = 0
    maximum: float = 100
    step: float = 1
    decimals: int = 2
    choices: tuple = ()
    suffix: str = ""


@dataclass
class OpResult:
    image: np.ndarray
    log: str = ""
    kernel: np.ndarray | None = None
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Operation:
    key: str
    name: str
    category: str
    description: str
    func: Callable[[np.ndarray, dict], OpResult]
    params: tuple[Param, ...] = ()
    pointwise: bool = False


def _r(img, log: str = "", kernel=None, aux=None, **meta) -> OpResult:
    return OpResult(as_uint8(img), log=log, kernel=kernel,
                    aux=aux or {}, meta=meta)


def _positive_map(values) -> np.ndarray:
    v = np.where(np.isfinite(values), values, 0.0)
    peak = float(v.max())
    if peak <= 0:
        return np.zeros(v.shape, dtype=np.uint8)
    return as_uint8(np.clip(v, 0, peak) / peak * 255.0)


# ---------------------------------------------------------------------------
# foundations -- colour
# ---------------------------------------------------------------------------

def _op_gray_weighted(img, p):
    out = to_gray(img, "weighted")
    return _r(out, "I_gray = 0.30 R + 0.59 G + 0.11 B\n"
                   "Weighted method: matches perceived luminance "
                   "(most sensitive to green).")


def _op_gray_average(img, p):
    out = to_gray(img, "average")
    return _r(out, "I_gray = (R + G + B) / 3\nAverage method.")


def _op_channel(img, p):
    name = p["channel"]
    if img.ndim == 2:
        return _r(img, "Image is already grayscale.")
    idx = {"Red": 0, "Green": 1, "Blue": 2}[name]
    return _r(img[..., idx], f"Extracted the {name.lower()} channel "
                             f"(BGR order in OpenCV, RGB here).")


# ---------------------------------------------------------------------------
# point processing (notes 00)
# ---------------------------------------------------------------------------

def _op_negative(img, p):
    return _r(en.negative(img), "s = (L - 1) - r = 255 - r\n"
                                "Flips dark and light; useful for white/gray "
                                "detail on a dark background (X-rays).")


def _op_log(img, p):
    c = float(p["c"])
    out = en.log_transform(img, c if c > 0 else None)
    used = c if c > 0 else 255.0 / np.log1p(max(float(as_float(img).max()), 1.0))
    return _r(out, f"s = c · log(1 + r),  c = {used:.4f}\n"
                    "Expands dark values, compresses bright values.\n"
                    "c = 0 means automatic: c = 255 / log(1 + r_max).")


def _op_gamma(img, p):
    g = float(p["gamma"])
    c = float(p["c"])
    out = en.gamma_transform(img, g, c)
    effect = "brightens (expands dark values)" if g < 1 else (
        "is identity" if abs(g - 1) < 1e-9 else "darkens (compresses dark values)")
    return _r(out, f"s = c · r^γ with r = pixel/255, then scaled back to 0..255\n"
                    f"γ = {g:g}, c = {c:g}  ->  {effect}.")


def _op_piecewise(img, p):
    out = en.piecewise_linear(img, p["r1"], p["s1"], p["r2"], p["s2"])
    return _r(out,
              f"Control points: (r1,s1) = ({p['r1']}, {p['s1']}), "
              f"(r2,s2) = ({p['r2']}, {p['s2']}).\n"
              "Three linear pieces; a steep middle segment (slope > 1) "
              "stretches the mid-range intensities and boosts contrast.")


def _op_contrast(img, p):
    a = as_float(img)
    lo_pct, hi_pct = float(p["low"]), float(p["high"])
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    out = en.contrast_stretch(img, lo_pct, hi_pct)
    return _r(out, f"X_new = (X - Xmin) / (Xmax - Xmin) · 255\n"
                   f"input range [{lo:.2f}, {hi:.2f}] -> [0, 255] "
                   f"(percentiles {lo_pct:g} / {hi_pct:g}).\n"
                   "Percentile clipping protects the mapping from a single "
                   "outlier pixel.")


def _op_equalize(img, p):
    out = en.histogram_equalization(img)
    return _r(out, "s_k = round((L - 1) · CDF(k))\n"
                   "The CDF spreads dense intensity ranges and compresses "
                   "sparse ones. Discrete rounding means the output histogram "
                   "is only approximately uniform.\n\n" +
                   en.equalization_report(img))


# ---------------------------------------------------------------------------
# noise generators (for demonstrating the filters)
# ---------------------------------------------------------------------------

def _op_salt(img, p):
    n = p["amount"]
    out = sp.add_salt_and_pepper(img, n, p["seed"])
    return _r(out, f"Salt-and-pepper: {n:g}% of pixels set to 0 (pepper) "
                    f"or 255 (salt), seed = {p['seed']}.\n"
                    "Median filtering is the classic cure.")


def _op_gauss_noise(img, p):
    s = float(p["sigma"])
    out = sp.add_gaussian_noise(img, s, p["seed"])
    return _r(out, f"g = f + η,  η ~ N(0, σ²),  σ = {s:g}, seed = {p['seed']}.\n"
                    "Averaging N pixels cuts the noise variance to about σ²/N, "
                    "but it blurs edges too.")


# ---------------------------------------------------------------------------
# smoothing (notes 01)
# ---------------------------------------------------------------------------

def _op_box(img, p):
    k = int(p["k"])
    border = p["border"]
    kern = np.ones((k, k)) / (k * k)
    return _r(sp.box_average(img, k, border),
              f"Uniform average, {k}x{k}, border = {border}.\n"
              f"{describe_kernel(kern)}\n"
              "Smoothing kernels sum to 1, so overall brightness is preserved.\n"
              "Averaging N pixels reduces independent noise variance ~ N times.",
              kernel=kern)


def _op_weighted(img, p):
    border = p["border"]
    return _r(sp.weighted_average(img, border),
              "Weighted average, classic 1/16 mask, border = "
              f"{border}.\n{describe_kernel(sp.WEIGHTED3)}\n"
              "Centre 4, orthogonal 2, diagonal 1. The centre dominates, so it "
              "blurs less than uniform averaging. 16 = 2^4 allows a bit-shift.",
              kernel=sp.WEIGHTED3)


def _op_median(img, p):
    k, border = int(p["k"]), p["border"]
    return _r(sp.median_filter(img, k, border),
              f"Median filter {k}x{k}, border = {border}.\n"
              "Sort the neighbourhood and take the middle value. Excellent for "
              "salt-and-pepper noise, with less edge blur than linear filters.")


def _op_min(img, p):
    k, border = int(p["k"]), p["border"]
    return _r(sp.min_filter(img, k, border),
              f"Min filter {k}x{k} (rank 1), border = {border}.\n"
              "Keeps the darkest value -> removes bright specks.")


def _op_max(img, p):
    k, border = int(p["k"]), p["border"]
    return _r(sp.max_filter(img, k, border),
              f"Max filter {k}x{k} (rank k²), border = {border}.\n"
              "Keeps the brightest value -> removes dark specks.")


def _op_nth(img, p):
    k, rank = int(p["k"]), int(p["rank"])
    rank = int(np.clip(rank, 1, k * k))
    border = p["border"]
    meaning = ("minimum" if rank == 1 else
               "maximum" if rank == k * k else
               "median" if rank == (k * k + 1) // 2 else "nth order")
    return _r(sp.nth_order_filter(img, k, rank, border),
              f"{k}x{k} order-statistic filter, rank {rank} of {k * k} "
              f"({meaning}), border = {border}.\n"
              "min = 1st, max = last, median = middle.")


def _op_adaptive(img, p):
    noise_std = float(p["noise_std"])
    k, border = int(p["k"]), p["border"]
    out = sp.adaptive_local(img, noise_std, k, border)
    h, w = as_float(img).shape[:2]
    stats = sp.adaptive_stats(img, noise_std, k, h // 2, w // 2)
    return _r(out,
              "f̂ = g - (σ_η² / σ_L²) · (g - m_L)\n"
              f"window {k}x{k}, σ_η = {noise_std:g} (σ_η² = {noise_std ** 2:g}), "
              f"border = {border}.\n"
              "ratio ≈ 0 on an edge -> keep the pixel; ratio = 1 in a flat "
              "region -> replace by the local mean. Clamped at 1.\n" + stats)


# ---------------------------------------------------------------------------
# sharpening / edges (notes 01)
# ---------------------------------------------------------------------------

def _op_sobel(img, p):
    variant = p["variant"].lower()
    thr = float(p["threshold"])
    out = sp.sobel_magnitude(img, thr, variant)
    _, _, mag, _ = sp.sobel_gradients(img, variant)
    orient = sp.sobel_orientation(img, variant)
    kx = sp.SCHARR_X if variant == "scharr" else sp.SOBEL_X
    return _r(out,
              f"{p['variant']} gradient magnitude: G = √(Gx² + Gy²)\n"
              f"max gradient = {float(mag.max()):.2f}, "
              f"threshold = {thr:g} (0 = keep everything).\n"
              "Edges are bright on a dark background; flat regions give 0.\n"
              "Derivative masks sum to 0, so constant regions vanish.\n"
              "The panel shows the Gx mask; Gy is its transpose.",
              kernel=kx,
              aux={"Gradient direction": orient})


def _op_sobel_dir(img, p):
    variant = p["variant"].lower()
    out = sp.sobel_orientation(img, variant)
    mag = sp.sobel_magnitude(img, 0, variant)
    return _r(out,
              f"{p['variant']} gradient direction θ = atan2(Gy, Gx).\n"
              "Hue encodes direction, brightness encodes gradient strength.",
              aux={"Gradient magnitude": mag})


def _op_laplacian(img, p):
    kind = p["kind"]
    out = sp.laplacian_display(img, kind)
    k = sp.LAPLACIANS[kind]
    return _r(out,
              f"Laplacian, {kind} mask. |∇²f| of the luminance displayed.\n"
              f"{describe_kernel(k)}\n"
              "∇²f = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y).\n"
              "Sum of coefficients is 0 -> flat regions give 0.",
              kernel=k)


def _op_laplacian_sharp(img, p):
    kind = p["kind"]
    strength = float(p["strength"])
    out = sp.laplacian_sharpen(img, kind, strength)
    return _r(out,
              f"g = f - {strength:g} · ∇²f  ({kind} Laplacian mask).\n"
              "The centre coefficient is negative, so the response is "
              "subtracted. The output is clipped to 0..255.",
              kernel=sp.LAPLACIANS[kind],
              aux={"Laplacian response": sp.laplacian_display(img, kind)})


# ---------------------------------------------------------------------------
# frequency domain (notes 02)
# ---------------------------------------------------------------------------

def _freq_op(img, p, mode: str) -> OpResult:
    kind = p["kind"]
    d0 = float(p["d0"])
    order = float(p["order"])
    pad = float(p["padding"])

    if mode == "low":
        mask_fn = lambda shape: fr.lowpass_mask(shape, d0, order, kind)
        label = "low-pass (smoothing / blurring)"
    else:
        mask_fn = lambda shape: fr.highpass_mask(shape, d0, order, kind)
        label = "high-pass (sharpening / edges)"

    out, mask, spec_in, spec_out = fr.frequency_filter(img, mask_fn, pad)
    kept = float(mask.mean()) * 100.0

    if kind == "ideal":
        formula = ("H = 1 if D(u,v) ≤ D0 else 0"
                   + ("  (low-pass)" if mode == "low" else "  (high-pass)"))
    elif kind == "butterworth":
        formula = (f"H = 1 / (1 + [D/D0]^(2n))  with n = {order:g}"
                   if mode == "low" else
                   f"H = 1 / (1 + [D0/D]^(2n))  with n = {order:g}")
    else:
        formula = ("H = exp(-D² / 2D0²)" if mode == "low"
                   else "H = 1 - exp(-D² / 2D0²)")

    return _r(out,
              f"{p['kind']} {label}\n{formula}\n"
              f"D0 = {d0:g}, padding factor = {pad:g}, "
              f"{kept:.1f}% of frequencies kept.\n"
              "Pipeline: f -> DFT -> multiply by H -> inverse DFT -> g.\n"
              "Ideal filters ring; Gaussian filters do not.",
              aux={"Input |F|": spec_in,
                   "Filter H": fr.mask_uint8(mask),
                   "Output |G|": spec_out})


def _op_lp(img, p):
    return _freq_op(img, p, "low")


def _op_hp(img, p):
    res = _freq_op(img, p, "high")
    if p.get("rescale", True):
        res.image = normalize_to_uint8(res.image)
        res.log += ("\nDisplay: output linearly rescaled to the full 0..255 "
                    "range (high-pass results are naturally dim).")
    return res


def _op_spectrum(img, p):
    pad = float(p["padding"])
    spec, F = fr.spectrum_of(img, pad)
    p_, q_ = F.shape
    return _r(spec,
              f"Centred Fourier spectrum, log(1 + |F|), padded to {p_}x{q_}.\n"
              "Centre = DC component (average brightness).\n"
              "Near the centre = low frequencies (smooth regions);\n"
              "far from the centre = high frequencies (edges, detail, noise).\n"
              "The transform is lossless: the inverse recovers the image.")


def _op_band(img, p):
    reject = bool(p["reject"])
    kind = p["kind"]
    d0, d1 = float(p["d0"]), float(p["d1"])
    order, pad = float(p["order"]), float(p["padding"])
    mask_fn = lambda shape: fr.band_mask(shape, d0, d1, order, kind, reject)
    out, mask, spec_in, spec_out = fr.frequency_filter(img, mask_fn, pad)
    name = "band-reject" if reject else "band-pass"
    return _r(out,
              f"{p['kind']} {name}: keep D0 ≤ D ≤ D1 "
              f"({d0:g} .. {d1:g}), order = {order:g}.\n"
              "Band-pass = low-pass(D1) · high-pass(D0).\n"
              "Band-reject (notch) is the complement and is the tool for "
              "periodic noise.",
              aux={"Input |F|": spec_in,
                   "Filter H": fr.mask_uint8(mask),
                   "Output |G|": spec_out})


def _op_notch(img, p):
    du, dv = float(p["du"]), float(p["dv"])
    radius, pad = float(p["radius"]), float(p["padding"])
    mask_fn = lambda shape: fr.notch_reject_mask(shape, du, dv, radius)
    out, mask, spec_in, spec_out = fr.frequency_filter(img, mask_fn, pad)
    return _r(out,
              f"Notch reject around offset (du, dv) = ({du:g}, {dv:g}) and its "
              f"mirror, radius = {radius:g}.\n"
              "Periodic noise appears as isolated bright spikes in the "
              "spectrum; zeroing both symmetric spikes removes it.",
              aux={"Input |F|": spec_in,
                   "Filter H": fr.mask_uint8(mask),
                   "Output |G|": spec_out})


def _op_homomorphic(img, p):
    gl, gh = float(p["gamma_l"]), float(p["gamma_h"])
    d0, pad = float(p["d0"]), float(p["padding"])
    out = fr.homomorphic(img, gl, gh, d0, pad)

    a = as_float(img)
    H, W = a.shape[:2]
    P, Q = fr.padded_size(H, W, pad)
    D = fr.distance_grid((P, Q))
    Hf = ((gh - gl) * (1.0 - np.exp(-(D * D) / (2.0 * max(d0, EPS) ** 2)))
          + gl)
    spec, _ = fr.spectrum_of(img, pad)

    return _r(out,
              "f = i · r  ->  log  ->  DFT  ->  H  ->  IDFT  ->  exp  ->  g\n"
              f"γ_L = {gl:g} (illumination gain, < 1 compresses dynamic range), "
              f"γ_H = {gh:g} (reflectance gain, > 1 boosts detail), D0 = {d0:g}.\n"
              "H = (γ_H - γ_L)(1 - exp(-D²/2D0²)) + γ_L.\n"
              "Output is rescaled (2nd..98th percentile) to 0..255 for display.",
              aux={"Input spectrum": spec,
                   "Filter H": fr.mask_uint8(Hf)})


# ---------------------------------------------------------------------------
# features (notes 03)
# ---------------------------------------------------------------------------

def _op_harris(img, p):
    k = float(p["k"])
    window = int(p["window"])
    pct = float(p["threshold"])
    md = int(p["min_distance"])
    radius = int(p["radius"])

    points, r = ft.harris_corners(img, k, window, pct, md)
    vis = draw_markers(img, points, radius=radius)
    return _r(vis,
              "R = det(M) - k · trace(M)² = λ1λ2 - k(λ1 + λ2)²\n"
              f"k = {k:g}, window = {window}x{window}, threshold = "
              f"{pct:g}th percentile, min distance = {md} px.\n"
              f"corners found: {len(points)}\n"
              "Corner: both eigenvalues large -> R positive.\n"
              "Edge: one large, one small -> R negative. "
              "Flat: both small -> R ≈ 0.",
              aux={"Harris response R": _positive_map(r)})


def _op_harris_map(img, p):
    k = float(p["k"])
    window = int(p["window"])
    r = ft.harris_response(img, k, window)
    return _r(_positive_map(r),
              "Harris response R = det(M) - k·trace(M)², positive part "
              "rescaled to 0..255.\nBright = corner-like, dark = flat or edge.",
              aux={})


def _op_hog(img, p):
    cell = int(p["cell"])
    bins = int(p["bins"])
    block = int(p["block"])
    stride = int(p["stride"])
    resize = (64, 128) if p["resize"].startswith("64") else None

    d = ft.hog(img, cell, bins, block, stride, resize)
    scale = int(np.clip(440 // max(d["cells"][1], 1), 6, 26))
    viz = ft.hog_visualization(d["hist"], bins, scale)

    ny, nx = d["cells"]
    by, bx = d["blocks"]
    length = by * bx * (block * block) * bins
    mag = d["mag"]
    mag_img = normalize_to_uint8(mag, 0.0, max(float(mag.max()), EPS))

    extra = f"\n{d['color_note']}" if d["color_note"] else ""
    return _r(viz,
              f"window {d['size'][1]}x{d['size'][0]} px, cell {cell}x{cell}, "
              f"{bins} bins, block {block}x{block} cells, stride {stride}.\n"
              f"cells: {nx} x {ny} (={nx * ny}), "
              f"block positions: {bx} x {by} (={bx * by}).\n"
              f"descriptor length = {bx}·{by}·({block}²·{bins}) = {length}.\n"
              "9 bins = 20° each; votes split softly between the two nearest "
              "bins; blocks are L2-normalised to cancel contrast changes."
              + extra,
              aux={"Gradient magnitude": mag_img})


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "Foundations",
    "Point Processing",
    "Noise",
    "Smoothing",
    "Sharpening",
    "Frequency Domain",
    "Features",
]

_BORDER = Param("border", "Border padding", "choice", "reflect",
                choices=("reflect", "replicate", "zeros"))

_PAD = Param("padding", "FFT padding", "choice", "1",
             choices=("1", "2"))

_KIND = Param("kind", "Filter type", "choice", "butterworth",
              choices=("ideal", "butterworth", "gaussian"))

OPERATIONS: tuple[Operation, ...] = (
    # ---- foundations ------------------------------------------------------
    Operation("gray_weighted", "Grayscale (weighted)", "Foundations",
              "RGB -> grayscale with the perceptual weights "
              "0.30 R + 0.59 G + 0.11 B (the eye is most sensitive to green).",
              _op_gray_weighted),
    Operation("gray_average", "Grayscale (average)", "Foundations",
              "RGB -> grayscale with the plain average (R + G + B) / 3.",
              _op_gray_average),
    Operation("channel", "Extract colour channel", "Foundations",
              "Split the image into one of its three intensity planes. "
              "A colour image is three stacked matrices.",
              _op_channel,
              (Param("channel", "Channel", "choice", "Red",
                     choices=("Red", "Green", "Blue")),)),

    # ---- point processing -------------------------------------------------
    Operation("negative", "Negative", "Point Processing",
              "s = (L - 1) - r. Reverses brightness; useful when detail is "
              "white/gray on a dark background.",
              _op_negative, pointwise=True),
    Operation("log", "Logarithmic", "Point Processing",
              "s = c · log(1 + r). Expands dark values and compresses bright "
              "ones; also used to display Fourier spectra.",
              _op_log,
              (Param("c", "c (0 = auto)", "float", 0.0, 0, 10, 0.05, 3),),
              pointwise=True),
    Operation("gamma", "Power-law (gamma)", "Point Processing",
              "s = c · r^γ after normalising r to 0..1. γ < 1 brightens, "
              "γ > 1 darkens.",
              _op_gamma,
              (Param("gamma", "γ", "float", 0.6, 0.05, 5.0, 0.05, 2),
               Param("c", "c", "float", 1.0, 0, 3, 0.05, 2)),
              pointwise=True),
    Operation("piecewise", "Piecewise linear", "Point Processing",
              "Three linear segments through control points (r1,s1) and "
              "(r2,s2). A steep middle segment stretches mid-range contrast.",
              _op_piecewise,
              (Param("r1", "r1", "int", 70, 1, 250, 1),
               Param("s1", "s1", "int", 30, 0, 255, 1),
               Param("r2", "r2", "int", 180, 2, 254, 1),
               Param("s2", "s2", "int", 225, 0, 255, 1)),
              pointwise=True),
    Operation("contrast", "Contrast stretching", "Point Processing",
              "X_new = (X - Xmin)/(Xmax - Xmin) · 255. Scales the histogram "
              "wider; percentile clipping avoids outliers ruining the range.",
              _op_contrast,
              (Param("low", "Low percentile", "float", 0, 0, 49, 0.5, 1),
               Param("high", "High percentile", "float", 100, 51, 100, 0.5, 1))),
    Operation("equalize", "Histogram equalization", "Point Processing",
              "Remap intensities through the CDF: s_k = round((L-1)·CDF(k)). "
              "Reshapes the histogram towards uniform -> maximum contrast.",
              _op_equalize),

    # ---- noise ------------------------------------------------------------
    Operation("salt", "Add salt & pepper noise", "Noise",
              "Random pixels are forced to 0 (pepper) or 255 (salt).",
              _op_salt,
              (Param("amount", "Noise %", "float", 3, 0, 30, 0.5, 1),
               Param("seed", "Seed", "int", 0, 0, 9999, 1))),
    Operation("gauss_noise", "Add Gaussian noise", "Noise",
              "g = f + η with η ~ N(0, σ²). Larger σ means a noisier image.",
              _op_gauss_noise,
              (Param("sigma", "σ", "float", 15, 0, 100, 1, 1),
               Param("seed", "Seed", "int", 0, 0, 9999, 1))),

    # ---- smoothing --------------------------------------------------------
    Operation("box", "Average (box) filter", "Smoothing",
              "Uniform k x k average. Sum of coefficients = 1, so brightness "
              "is preserved. Bigger kernel = more denoising and more blur.",
              _op_box,
              (Param("k", "Kernel size", "int", 3, 3, 31, 2), _BORDER)),
    Operation("weighted", "Weighted average 1/16", "Smoothing",
              "(1/16)·[[1,2,1],[2,4,2],[1,2,1]]. The centre counts 4x, so it "
              "blurs less than a uniform average.",
              _op_weighted, (_BORDER,)),
    Operation("median", "Median filter", "Smoothing",
              "Middle value of the sorted neighbourhood. Removes "
              "salt-and-pepper noise with less edge blur.",
              _op_median,
              (Param("k", "Kernel size", "int", 3, 3, 15, 2), _BORDER)),
    Operation("min", "Min filter", "Smoothing",
              "Smallest value in the neighbourhood (rank 1).",
              _op_min,
              (Param("k", "Kernel size", "int", 3, 3, 15, 2), _BORDER)),
    Operation("max", "Max filter", "Smoothing",
              "Largest value in the neighbourhood (rank k²).",
              _op_max,
              (Param("k", "Kernel size", "int", 3, 3, 15, 2), _BORDER)),
    Operation("nth", "Nth-order statistic", "Smoothing",
              "Pick the n-th value after sorting the neighbourhood "
              "(1 = min, middle = median, k² = max).",
              _op_nth,
              (Param("k", "Kernel size", "int", 3, 3, 15, 2),
               Param("rank", "Rank (1..k²)", "int", 5, 1, 225, 1), _BORDER)),
    Operation("adaptive", "Adaptive local filter", "Smoothing",
              "f̂ = g - (σ_η²/σ_L²)(g - m_L). Smooths flat regions fully but "
              "keeps edges because the local variance is large there.",
              _op_adaptive,
              (Param("noise_std", "Noise σ_η", "float", 3.0, 0, 60, 0.5, 1),
               Param("k", "Window size", "int", 3, 3, 15, 2), _BORDER)),

    # ---- sharpening -------------------------------------------------------
    Operation("sobel", "Sobel magnitude", "Sharpening",
              "First derivative: G = √(Gx² + Gy²). Bright edges on black; "
              "flat regions respond with 0.",
              _op_sobel,
              (Param("variant", "Operator", "choice", "Sobel",
                     choices=("Sobel", "Scharr")),
               Param("threshold", "Threshold", "float", 25, 0, 255, 1, 0))),
    Operation("sobel_dir", "Sobel orientation", "Sharpening",
              "θ = atan2(Gy, Gx). Hue shows the gradient direction, "
              "brightness shows its magnitude.",
              _op_sobel_dir,
              (Param("variant", "Operator", "choice", "Sobel",
                     choices=("Sobel", "Scharr")),)),
    Operation("laplacian", "Laplacian response", "Sharpening",
              "Second derivative |∇²f|. Responds to edges and isolated "
              "points; coefficients always sum to zero.",
              _op_laplacian,
              (Param("kind", "Mask", "choice", "4-neighbor",
                     choices=("4-neighbor", "8-neighbor", "diagonal")),)),
    Operation("laplacian_sharp", "Laplacian sharpening", "Sharpening",
              "g = f - c·∇²f for a negative-centre mask, then clip to 0..255.",
              _op_laplacian_sharp,
              (Param("kind", "Mask", "choice", "4-neighbor",
                     choices=("4-neighbor", "8-neighbor", "diagonal")),
               Param("strength", "Strength c", "float", 1.0, 0, 4, 0.1, 1))),

    # ---- frequency --------------------------------------------------------
    Operation("spectrum", "Fourier spectrum", "Frequency Domain",
              "Centred log-magnitude spectrum. Centre = DC/average "
              "brightness, outside = edges, detail and noise.",
              _op_spectrum, (_PAD,)),
    Operation("lowpass", "Low-pass filter", "Frequency Domain",
              "Keep the centre, remove the outside: smooths/blurs the image. "
              "Ideal rings, Butterworth rolls off, Gaussian is smoothest.",
              _op_lp,
              (_KIND,
               Param("d0", "Cutoff D0", "float", 30, 1, 400, 1, 0),
               Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
               _PAD)),
    Operation("highpass", "High-pass filter", "Frequency Domain",
              "Remove the centre, keep edges. The DC term is zeroed, so smooth "
              "regions go black and only outlines remain.",
              _op_hp,
              (_KIND,
               Param("d0", "Cutoff D0", "float", 30, 1, 400, 1, 0),
               Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
               Param("rescale", "Rescale for display", "bool", True),
               _PAD)),
    Operation("bandpass", "Band-pass filter", "Frequency Domain",
              "Keep only D0 ≤ D ≤ D1. Suppresses low frequencies (edges) and "
              "high frequencies (noise) at the same time.",
              _op_band,
              (_KIND,
               Param("d0", "Lower D0", "float", 15, 1, 400, 1, 0),
               Param("d1", "Upper D1", "float", 55, 2, 500, 1, 0),
               Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
               Param("reject", "Band-reject", "bool", False),
               _PAD)),
    Operation("bandreject", "Band-reject (notch band)", "Frequency Domain",
              "Remove a ring of frequencies instead of keeping it. Used for "
              "periodic noise.",
              _op_band,
              (_KIND,
               Param("d0", "Lower D0", "float", 15, 1, 400, 1, 0),
               Param("d1", "Upper D1", "float", 55, 2, 500, 1, 0),
               Param("order", "Order n", "float", 2, 0.5, 10, 0.5, 1),
               Param("reject", "Band-reject", "bool", True),
               _PAD)),
    Operation("notch", "Notch reject (spikes)", "Frequency Domain",
              "Zero two symmetric spikes in the spectrum. Periodic noise "
              "produces exactly such isolated spikes.",
              _op_notch,
              (Param("du", "Offset du", "float", 40, -400, 400, 1, 0),
               Param("dv", "Offset dv", "float", 40, -400, 400, 1, 0),
               Param("radius", "Radius", "float", 6, 1, 80, 1, 0),
               _PAD)),
    Operation("homomorphic", "Homomorphic filtering", "Frequency Domain",
              "Compress illumination and boost reflectance at once: "
              "log -> DFT -> H -> IDFT -> exp.",
              _op_homomorphic,
              (Param("gamma_l", "γ_L (illumination)", "float", 0.5, 0.05, 1.0, 0.05, 2),
               Param("gamma_h", "γ_H (reflectance)", "float", 2.0, 1.0, 4.0, 0.1, 1),
               Param("d0", "Cutoff D0", "float", 30, 1, 400, 1, 0),
               _PAD)),

    # ---- features ---------------------------------------------------------
    Operation("harris", "Harris corners", "Features",
              "Detect corners: R = det(M) - k·trace(M)². Corners have two "
              "large eigenvalues; edges have one large and one small.",
              _op_harris,
              (Param("k", "k", "float", 0.04, 0.01, 0.10, 0.005, 3),
               Param("window", "Window", "int", 3, 3, 11, 2),
               Param("threshold", "Threshold percentile", "float", 99.8, 80, 99.9, 0.1, 1),
               Param("min_distance", "Min distance (px)", "int", 8, 1, 40, 1),
               Param("radius", "Marker radius", "int", 5, 2, 20, 1))),
    Operation("harris_map", "Harris response map", "Features",
              "The raw R map: bright pixels are corner candidates.",
              _op_harris_map,
              (Param("k", "k", "float", 0.04, 0.01, 0.10, 0.005, 3),
               Param("window", "Window", "int", 3, 3, 11, 2))),
    Operation("hog", "HOG descriptor", "Features",
              "Histogram of Oriented Gradients: gradients -> 8x8 cell "
              "histograms -> overlapping 2x2 block normalisation.",
              _op_hog,
              (Param("resize", "Resize first", "choice", "64 x 128",
                     choices=("64 x 128", "keep size")),
               Param("cell", "Cell size", "int", 8, 4, 32, 4),
               Param("bins", "Orientation bins", "int", 9, 4, 18, 1),
               Param("block", "Block (cells)", "int", 2, 1, 4, 1),
               Param("stride", "Block stride", "int", 1, 1, 3, 1))),
)


def operations_by_category() -> dict[str, list[Operation]]:
    grouped: dict[str, list[Operation]] = {c: [] for c in CATEGORY_ORDER}
    for op in OPERATIONS:
        grouped.setdefault(op.category, []).append(op)
    return grouped


def find_operation(key: str) -> Operation | None:
    for op in OPERATIONS:
        if op.key == key:
            return op
    return None
