#!/usr/bin/env python3
"""Self-test: verify the algorithms against the notes' worked examples.

Run:  python selftest.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from diplab.core import transforms as fr
from diplab.core.dft_reference import dft_1d, dft_2d, idft_1d, idft_2d
from diplab.core.utils import resize, to_gray
from diplab.features import harris as harris_mod
from diplab.features import hog as hog_mod
from diplab.point_processing import histogram_equalization as he
from diplab.frequency import common as fc
from diplab.operations import OPERATIONS, find_operation
from diplab.sharpening.common import LAPLACIANS, gradients

PASS, FAIL = 0, 0


def check(name: str):
    def deco(fn):
        global PASS, FAIL
        try:
            fn()
            print(f"  PASS  {name}")
            PASS += 1
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc(limit=2)
            FAIL += 1
        return fn
    return deco


def run_op(key: str, img: np.ndarray, **overrides):
    """Run a registry operation with its GUI defaults plus overrides."""
    op = find_operation(key)
    if op is None:
        raise KeyError(key)
    params = {p.key: p.default for p in op.params}
    params.update(overrides)
    return op.func(img, params)


# ---------------------------------------------------------------------------
print("FFT / DFT")
# ---------------------------------------------------------------------------

@check("FFT wrappers match numpy on random 1-D data")
def _():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    assert np.allclose(fr.fft(x), np.fft.fft(x), atol=1e-8)
    assert np.allclose(fr.ifft(x), np.fft.ifft(x), atol=1e-8)


@check("FFT wrappers match numpy on random 2-D data")
def _():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 32))
    assert np.allclose(fr.fft2(x), np.fft.fft2(x), atol=1e-7)
    assert np.allclose(fr.ifft2(fr.fft2(x)), x, atol=1e-9)


@check("FFT operates along an arbitrary axis")
def _():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 16, 4)) + 1j * rng.normal(size=(8, 16, 4))
    assert np.allclose(fr.fft(x, axis=1), np.fft.fft(x, axis=1), atol=1e-8)


@check("DFT example 4.1: [1,2,4,4] -> [11, -3+2j, -1, -3-2j]")
def _():
    f = np.array([1, 2, 4, 4], float)
    F = dft_1d(f)
    expected = np.array([11, -3 + 2j, -1, -3 - 2j])
    assert np.allclose(F, expected), F
    assert np.allclose(idft_1d(F), f)
    assert np.allclose(F[3], np.conj(F[1]))


@check("naive 2-D DFT matches the FFT")
def _():
    rng = np.random.default_rng(3)
    img = rng.normal(size=(8, 8))
    assert np.allclose(dft_2d(img), fr.fft2(img), atol=1e-8)
    assert np.allclose(idft_2d(dft_2d(img)), img, atol=1e-9)


@check("2-D example: F(0,0)/MN = 10 for the note's 3x2 image")
def _():
    img = np.array([[5, 10, 20], [5, 15, 5]], float)
    F = dft_2d(img)
    assert np.isclose(F[0, 0].real / img.size, 10.0)
    assert np.isclose(F[0, 0].real, img.sum())


# ---------------------------------------------------------------------------
print("\nEnhancement (notes 00)")
# ---------------------------------------------------------------------------

@check("weighted grayscale uses 0.3 / 0.59 / 0.11")
def _():
    rgb = np.zeros((1, 1, 3), np.uint8)
    rgb[0, 0] = (100, 150, 200)
    expected = 0.3 * 100 + 0.59 * 150 + 0.11 * 200
    assert np.isclose(to_gray(rgb)[0, 0], expected)


@check("negative: s = 255 - r")
def _():
    a = np.array([[0, 100, 255]], np.uint8)
    assert np.array_equal(run_op("negative", a).image, [[255, 155, 0]])


@check("gamma = 1 is the identity (within rounding)")
def _():
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = run_op("gamma", a, gamma=1.0, c=1.0).image
    assert np.array_equal(out, a)


@check("gamma < 1 brightens, gamma > 1 darkens")
def _():
    a = np.full((4, 4), 64, np.uint8)
    assert run_op("gamma", a, gamma=0.5, c=1.0).image[0, 0] > 64
    assert run_op("gamma", a, gamma=2.0, c=1.0).image[0, 0] < 64


@check("log transform: monotonic, 0 -> 0, max -> 255 (auto c)")
def _():
    a = np.array([[0, 1, 50, 128, 255]], np.uint8)
    out = run_op("log", a).image[0].astype(float)
    assert out[0] == 0 and out[-1] == 255
    assert np.all(np.diff(out) >= 0)


@check("piecewise linear hits all control points")
def _():
    a = np.array([[0, 70, 180, 255]], np.uint8)
    out = run_op("piecewise", a, r1=70, s1=30, r2=180, s2=225).image[0]
    assert out[0] == 0 and out[-1] == 255
    assert abs(int(out[1]) - 30) <= 1 and abs(int(out[2]) - 225) <= 1


@check("contrast stretch maps min -> 0 and max -> 255")
def _():
    a = np.array([[50, 60, 100, 180]], np.uint8)
    out = run_op("contrast", a, low=0, high=100).image[0]
    assert out[0] == 0 and out[-1] == 255
    assert out[1] == 20 or out[1] == 19     # 10/130*255 = 19.6


@check("histogram equalization matches the note's 8-level example")
def _():
    counts = [70, 100, 40, 60, 10, 70, 10, 40]
    img = np.repeat(np.arange(8), counts).reshape(-1, 1)
    lut = he.equalization_lut(img, levels=8)
    expected = np.array([1, 3, 4, 5, 5, 6, 6, 7])
    assert np.array_equal(lut, expected), lut


@check("equalization output uses the full range")
def _():
    rng = np.random.default_rng(4)
    img = (rng.normal(100, 12, size=(64, 64))).clip(0, 255).astype(np.uint8)
    out = run_op("equalize", img).image
    assert out.max() >= 250 and out.min() <= 5


# ---------------------------------------------------------------------------
print("\nSpatial filtering (notes 01)")
# ---------------------------------------------------------------------------

NOTE_IMAGE = np.array([
    [12, 16, 18, 35, 30, 28],
    [8, 50, 20, 24, 30, 22],
    [2, 12, 3, 21, 34, 18],
    [5, 6, 4, 3, 32, 12],
    [10, 1, 0, 4, 15, 23],
], dtype=np.uint8)


@check("3x3 average around the spike 50 gives 16 (141/9)")
def _():
    out = run_op("box", NOTE_IMAGE, k=3).image
    assert out[1, 1] == 16, out[1, 1]


@check("weighted 1/16 average around the spike gives 22 (347/16)")
def _():
    out = run_op("weighted", NOTE_IMAGE).image
    assert out[1, 1] == 22, out[1, 1]


@check("median removes salt-and-pepper noise completely")
def _():
    base = np.full((64, 64), 120, np.uint8)
    noisy = run_op("salt", base, amount=10, seed=5).image
    out = run_op("median", noisy, k=3).image
    assert np.median(out) == 120
    bad = np.mean(out != 120)
    assert bad < 0.01, bad


@check("min / max / nth-order are consistent")
def _():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.uint8)
    # border = reflect makes the whole image a valid window at the centre
    assert run_op("min", a, k=3).image[1, 1] <= 1
    assert run_op("max", a, k=3).image[1, 1] >= 9
    assert run_op("nth", a, k=3, rank=5).image[1, 1] == 5


@check("adaptive filter example: 11 -> 9 with sigma_eta^2 = 2")
def _():
    window = np.array([[1, 3, 2], [1, 11, 1], [2, 4, 3]], np.uint8)
    out = run_op("adaptive", window, noise_std=np.sqrt(2.0), k=3).image
    assert out[1, 1] == 9, out[1, 1]


@check("adaptive filter returns flat noisy regions to the local mean")
def _():
    rng = np.random.default_rng(11)
    a = np.clip(80 + rng.normal(0, 3.0, size=(24, 24)), 0, 255).astype(np.uint8)
    out = run_op("adaptive", a, noise_std=3.0, k=3).image
    assert abs(float(out.mean()) - float(a.mean())) < 3.0
    assert float(out.std()) < float(a.std())


@check("Sobel responds to a vertical edge and is flat elsewhere")
def _():
    img = np.zeros((32, 32), np.uint8)
    img[:, 16:] = 220
    _, _, mag, _ = gradients(img)
    col = int(np.argmax(mag))
    assert abs((col % 32) - 16) <= 2, col
    assert mag[5, 5] < 1e-6


@check("Laplacian of a constant region is zero")
def _():
    img = np.full((16, 16), 77, np.uint8)
    lap = run_op("laplacian", img).image
    assert np.all(lap == 0)


@check("Laplacian sharpening keeps a constant image constant")
def _():
    img = np.full((16, 16), 90, np.uint8)
    out = run_op("laplacian_sharp", img).image
    assert np.all(out == 90)


@check("border modes produce the correct output shape")
def _():
    img = np.arange(25, dtype=np.uint8).reshape(5, 5)
    for border in ("reflect", "replicate", "zeros"):
        for k in (3, 5):
            assert run_op("box", img, k=k, border=border).image.shape == img.shape


# ---------------------------------------------------------------------------
print("\nFrequency domain (notes 02)")
# ---------------------------------------------------------------------------

@check("spectrum: DC is at the centre after fftshift")
def _():
    img = np.full((16, 16), 40, np.uint8)
    F = fr.fftshift2(fr.fft2(img))
    assert np.isclose(abs(F[8, 8]), 40 * 16 * 16, rtol=1e-6)


@check("Butterworth LP = 0.5 at D = D0 for every order")
def _():
    D = fr.distance_grid(32, 32)
    for n in (1, 2, 4):
        H = fc.lowpass_mask((32, 32), 5, n, "butterworth")
        at = H[np.isclose(D, 5, atol=0.01)]
        assert np.allclose(at, 0.5, atol=0.02), at


@check("Ideal LP is 1 at the centre and 0 far away; HP is its complement")
def _():
    shape = (32, 32)
    lp = fc.lowpass_mask(shape, 4, 2, "ideal")
    hp = fc.highpass_mask(shape, 4, 2, "ideal")
    assert lp[16, 16] == 1 and hp[16, 16] == 0
    assert lp[0, 0] == 0 and hp[0, 0] == 1
    assert np.allclose(lp + hp, 1)


@check("low-pass on a constant image preserves brightness")
def _():
    img = np.full((32, 32), 100, np.uint8)
    out, _, _, _ = fr.apply_filter(
        img, lambda s: fc.lowpass_mask(s, 6, 2, "gaussian"), 1.0)
    assert abs(float(out.mean()) - 100) < 1.0


@check("high-pass output of a constant image has mean ~ 0")
def _():
    img = np.full((32, 32), 100, np.uint8)
    out, _, _, _ = fr.apply_filter(
        img, lambda s: fc.highpass_mask(s, 6, 2, "gaussian"), 1.0)
    assert abs(float(out.mean())) < 1.0


@check("band masks are the product of LP and HP")
def _():
    shape = (32, 32)
    bp = fc.band_mask(shape, 3, 8, 2, "ideal")
    manual = (fc.lowpass_mask(shape, 8, 2, "ideal")
              * (1 - fc.lowpass_mask(shape, 3, 2, "ideal")))
    assert np.allclose(bp, manual)
    assert np.allclose(fc.band_mask(shape, 3, 8, 2, "ideal", True), 1 - manual)


@check("notch mask zeroes both symmetric spikes")
def _():
    P, Q = 32, 32
    m = fc.notch_reject_mask((P, Q), du=5, dv=3, radius=2)
    assert m[P // 2 + 3, Q // 2 + 5] == 0
    assert m[P // 2 - 3, Q // 2 - 5] == 0
    assert m[P // 2, Q // 2] == 1


@check("homomorphic filtering produces a finite image of the same size")
def _():
    rng = np.random.default_rng(6)
    img = (rng.normal(110, 25, size=(48, 48))).clip(0, 255).astype(np.uint8)
    out = run_op("homomorphic", img, gamma_l=0.5, gamma_h=2.0,
                 d0=10, padding="1").image
    assert out.shape == img.shape and out.dtype == np.uint8


# ---------------------------------------------------------------------------
print("\nFeatures (notes 03)")
# ---------------------------------------------------------------------------

@check("Harris finds the corners of a bright square")
def _():
    img = np.zeros((120, 120), np.uint8)
    img[35:85, 35:85] = 230
    points, _ = harris_mod.harris_corners(img, threshold_pct=97,
                                          min_distance=6)
    expected = [(35, 35), (35, 84), (84, 35), (84, 84)]
    hits = sum(any(abs(y - ey) <= 6 and abs(x - ex) <= 6 for y, x in points)
               for ey, ex in expected)
    assert hits >= 3, (hits, points)


@check("Harris response is ~0 on a flat image")
def _():
    r = harris_mod.harris_response(np.full((40, 40), 100, np.uint8))
    assert np.allclose(r, 0)


@check("HOG length is 3780 for 64x128 / cell 8 / 9 bins / 2x2 blocks")
def _():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(128, 64), dtype=np.uint8)
    d = hog_mod.compute(img, cell=8, bins=9, block=2, stride=1)
    assert d["cells"] == (16, 8)
    assert d["blocks"] == (15, 7)
    assert d["vector"].size == 3780, d["vector"].size


@check("HOG blocks are L2-normalised")
def _():
    rng = np.random.default_rng(8)
    img = rng.integers(0, 256, size=(128, 64), dtype=np.uint8)
    d = hog_mod.compute(img, 8, 9, 2, 1)
    blocks = d["vector"].reshape(-1, 36)
    norms = np.linalg.norm(blocks, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@check("HOG bins a vertical edge into orientation bin 0")
def _():
    img = np.zeros((16, 16), np.uint8)
    img[:, 8:] = 100        # vertical edge -> horizontal gradient (0 deg)
    d = hog_mod.compute(img, cell=16, bins=9, block=1, stride=1)
    h = d["hist"][0, 0]
    assert h.sum() > 0
    assert int(np.argmax(h)) == 0, h


# ---------------------------------------------------------------------------
print("\nUtils + integration")
# ---------------------------------------------------------------------------

@check("resize returns the requested shape")
def _():
    img = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    out = resize(img, 15, 10, "area")
    assert out.shape == (10, 15, 3)


@check("every operation runs on colour and grayscale input")
def _():
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    gray = to_gray(rgb).astype(np.uint8)
    failures = []
    for op in OPERATIONS:
        params = {p.key: p.default for p in op.params}
        for label, img in (("rgb", rgb), ("gray", gray)):
            try:
                res = op.func(img, params)
                if res.image.dtype != np.uint8:
                    failures.append(f"{op.key}/{label}: dtype {res.image.dtype}")
                if res.image.ndim not in (2, 3):
                    failures.append(f"{op.key}/{label}: ndim {res.image.ndim}")
                for name, aux in res.aux.items():
                    if aux.dtype != np.uint8:
                        failures.append(f"{op.key}/{label} aux {name}: "
                                        f"{aux.dtype}")
            except Exception as exc:              # noqa: BLE001
                failures.append(f"{op.key}/{label}: {type(exc).__name__}: {exc}")
    assert not failures, "\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
