# DIP Lab — Digital Image Processing, from scratch

A complete, dependency-light implementation of every algorithm in the four DIP
lecture notes, plus a desktop GUI (PyQt5) that loads an image, lists the
algorithms, exposes their parameters, and previews the result live.

All image processing is written **from scratch with NumPy** — there is no
OpenCV, no scipy, no scikit-image. Even the 2-D FFT, the median filter, the
histogram CDF and the HOG descriptor are implemented by hand.

```
python main.py
```

A synthetic demo scene is loaded on startup so the app is immediately usable.

![DIP Lab](docs/screenshot.png)

---

## Requirements

| Package | Version used | Purpose |
|---|---|---|
| Python | 3.10+ (tested on 3.14) | language |
| NumPy | 2.x | all numerical work |
| Pillow | 10+ | reading and writing image files only |
| PyQt5 | 5.15 | the GUI |

```
pip install numpy pillow PyQt5
```

Tkinter is **not** required.

---

## Project layout

```
main.py                     launch the GUI
selftest.py                 40 checks against the notes' worked examples
requirements.txt
diplab/
  utils.py                  dtype/channel helpers, padding, correlation,
                            integral-image box filter, resizing, drawing
  enhancement.py            point processing + histogram methods   (notes 00)
  spatial.py                smoothing, order statistics, adaptive,
                            Sobel/Scharr, Laplacian              (notes 01)
  frequency.py              DFT, radix-2 FFT, masks, homomorphic   (notes 02)
  features.py               Harris corners, HOG                    (notes 03)
  operations.py             GUI registry: 32 operations + parameters
  demo.py                   synthetic 512×512 test scene
  gui/
    widgets.py              zoomable image canvas, histogram, curve
    main_window.py          algorithm browser, live preview, controls
    style.py                dark stylesheet
```

---

## Using the GUI

| Area | What it does |
|---|---|
| **Left** | Searchable algorithm tree grouped by note. Click an algorithm to select it. |
| **Centre top** | Input and result images side by side. Scroll to zoom, drag to pan, double-click to fit. |
| **Centre bottom** | `Histogram` (input vs result), `Transfer curve` (for point operations), `Explanation` (formulas, parameters, timing), plus dynamic tabs for auxiliary images (spectrum, filter mask, Harris response, HOG magnification, …). |
| **Right** | The algorithm's theory blurb, its parameters, the kernel (for spatial filters), and the Apply button. |

**Auto preview** is on by default: moving a slider re-runs the operation after a
short delay. Turn it off if an operation is slow, then use **Apply** (F5).

**Working size** (toolbar) caps the processing resolution (default 1024 px,
`Full` for none). Frequency operations pad to the next power of two, so this
keeps the hand-written FFT responsive on large photos. The input view always
shows the working copy; **Reset input** goes back to the original file.

| Shortcut | Action |
|---|---|
| `Ctrl+O` | open image |
| `Ctrl+S` | save result |
| `Ctrl+R` | reset input to the original |
| `Ctrl+U` | use the result as the new input (chaining) |
| `F5` | apply (when auto preview is off) |
| drag & drop | drop an image file anywhere on the window |

`Tools → DFT worked example` reproduces the 1-D example from Lecture 4 inside
the app; `Tools → Algorithm map` lists coverage.

Images can be PNG, JPEG, BMP, TIFF, WebP or GIF, grayscale or colour. 16-bit
grayscale is rescaled to 8 bits on load. Colour images are processed per
channel unless the operation is explicitly a grayscale operation (Sobel,
Laplacian, Harris, HOG, spectrum).

---

## Algorithm map

### Notes 00 — Foundations & Enhancement (Lectures 1–2)

| Operation | Formula / idea |
|---|---|
| Grayscale (weighted) | `I = 0.30 R + 0.59 G + 0.11 B` |
| Grayscale (average) | `I = (R + G + B) / 3` |
| Extract colour channel | a colour image is three stacked matrices |
| Negative | `s = (L − 1) − r` |
| Logarithmic | `s = c · log(1 + r)`, auto `c = 255 / log(1 + r_max)` |
| Power-law (gamma) | `s = c · r^γ` with `r` normalised to 0..1 |
| Piecewise linear | three segments through `(r1,s1)` and `(r2,s2)` |
| Contrast stretching | `(X − X_min) / (X_max − X_min) · 255` with optional percentile clipping |
| Histogram equalization | `s_k = round((L − 1) · CDF(k))`, implemented as a 256-entry LUT |

The transfer curve panel plots `s = T(r)` by evaluating the operation on the
0..255 ramp, which is exactly the LUT concept from the notes.

> The equalization implementation deliberately uses the lecture's formula
> **without** subtracting `cdf_min`. This reproduces the worked 8-level example
> (`0→1, 1→3, 2→4, 3→5, 4→5, 5→6, 6→6, 7→7`) and is checked by the self-test.

### Notes 01 — Spatial Filters (Lecture 3)

| Operation | Formula / idea |
|---|---|
| Average (box) `k×k` | every coefficient `1/k²`; implemented with an integral image, O(HW) per pixel |
| Weighted average | `(1/16)·[[1,2,1],[2,4,2],[1,2,1]]` |
| Median | middle value of the sorted neighbourhood |
| Min / Max | rank 1 / rank `k²` |
| Nth-order statistic | rank `n` of `k²` (median is the middle rank) |
| Adaptive local filter | `f̂ = g − (σ_η²/σ_L²)(g − m_L)`, ratio clamped at 1 |
| Add salt & pepper | random pixels forced to 0 or 255 |
| Add Gaussian noise | `g = f + η`, `η ~ N(0, σ²)` |
| Sobel / Scharr magnitude | `G = √(Gx² + Gy²)`, optional threshold |
| Sobel / Scharr orientation | `θ = atan2(Gy, Gx)` encoded as hue |
| Laplacian response | `4`-neighbour, `8`-neighbour and diagonal masks |
| Laplacian sharpening | `g = f − c·∇²f` (all note masks have a negative centre) |

Implementation details that follow the notes:

* Correlation, not flipped convolution: `g(x,y) = Σ w(s,t)·f(x+s,y+t)`.
* Border modes: `reflect` is implemented as **symmetric** padding, because the
  note's `3 2 1 | 1 2 3 | 3 2 1` example is symmetric padding; `replicate` is
  edge padding and `zeros` is constant 0.
* Smoothing kernels sum to 1 (brightness preserved); derivative and Laplacian
  kernels sum to 0 (constant regions give 0).
* The self-test reproduces the note's numbers: average around 50 → 16,
  weighted → 22, adaptive with `σ_η² = 2` → 9.

### Notes 02 — Frequency Domain (Lecture 4)

| Operation | Formula / idea |
|---|---|
| Fourier spectrum | centred `log(1 + |F|)` display, DC in the middle |
| Low-pass | ideal, Butterworth `1/(1+(D/D0)^{2n})`, Gaussian `exp(−D²/2D0²)` |
| High-pass | complement `1 − H_low`, optional display rescale |
| Band-pass | `H_low(D1) · (1 − H_low(D0))` |
| Band-reject | complement of the band-pass |
| Notch reject | zero two symmetric discs around an offset spike |
| Homomorphic | `log → DFT → H → IDFT → exp`, `H = (γ_H−γ_L)(1−e^{−D²/2D0²}) + γ_L` |

Both transforms exist in the package:

* `dft_1d`, `idft_1d`, `dft_2d`, `idft_2d` — naive O(N²) versions, used for the
  worked example and as an independent check.
* `fft`, `ifft`, `fft2`, `ifft2` — an iterative radix-2 Cooley–Tukey FFT written
  from scratch and vectorised across all leading axes. Inputs are zero-padded
  to the next power of two; padding factor 2 is offered in the GUI to reduce
  wraparound.

The distance grid is `D(u,v) = √((u−P/2)² + (v−Q/2)²)` on the **centred**
spectrum, exactly as in the notes. The spectrum is shifted before multiplying
by `H` and shifted back before the inverse transform (this detail is what makes
the DC term line up with the centre of the mask).

The self-test verifies the FFT against `numpy.fft` for 1-D and 2-D data, checks
that `f = [1,2,4,4]` gives `F = [11, −3+2j, −1, −3−2j]`, that the 2-D example
has `F(0,0) = sum(f)`, and that the Butterworth response is 0.5 at `D = D0`.

### Notes 03 — Image Features (Lecture 5)

| Operation | Formula / idea |
|---|---|
| Harris corners | `M = Σ_w [[Ix², IxIy],[IxIy, Iy²]]`, `R = det(M) − k·trace(M)²`, 3×3 non-maximum suppression, percentile threshold, minimum-distance thinning |
| Harris response map | positive part of `R` rescaled to 0..255 |
| HOG descriptor | gradients `[-1,0,1]`, unsigned orientation folded to 0..180°, soft binning into 9 bins, 8×8 cells, overlapping 2×2-cell blocks with L2 normalisation, plus the classic star visualisation |

For a 64×128 window with 8×8 cells, 9 bins and 2×2 blocks with stride 1 the
descriptor length is

```
(15 block rows) × (7 block cols) × (2·2 cells × 9 bins) = 3780
```

which is asserted by the self-test. The GUI resizes to 64×128 by default
(matching the lecture's fixed window), but `keep size` is available.

---

## Self-test

```
python selftest.py
```

It prints one line per check and exits non-zero on failure. The 40 checks cover:

* radix-2 FFT vs `numpy.fft` (1-D, 2-D, arbitrary axis)
* the Lecture 4 DFT example `[1,2,4,4]`
* the 2-D DFT example `F(0,0)/MN = 10`
* the grayscale weights
* gamma/log/piecewise/contrast-stretch identities
* the Lecture 1 histogram-equalization mapping
* the Lecture 3 numeric examples (average 16, weighted 22, adaptive 9)
* border modes, median denoising, Sobel on a step edge, Laplacian of a flat field
* DC placement in the spectrum, Butterworth `−3 dB` point, band/notch masks
* homomorphic output sanity
* Harris corner locations on a synthetic square
* HOG length 3780 and per-block L2 norm
* a smoke test that runs **all 32 operations** on both RGB and grayscale input

Current result: **40 passed, 0 failed**.

---

## Notes and limitations

* Everything runs on the GUI thread. Frequency operations on a 1024 px image
  take roughly 1–3 seconds in pure NumPy; the working-size control exists to
  keep that interactive. Raise it to `Full` when exporting.
* The FFT requires power-of-two lengths, so images are zero-padded first. The
  spectra therefore describe the padded image; results are cropped back to the
  original size.
* Colour data is filtered per channel for linear operations. This is the
  straightforward extension of the grayscale theory and can shift saturation
  for non-linear operations such as per-channel equalization.
* Harris aggregates the structure tensor with a Gaussian window (σ = window/3);
  the response formula and non-maximum suppression follow the lecture.
* HOG uses unsigned gradients and L2 block normalisation as described in the
  notes. The optional L2-Hys clipping variant is not applied.
* The demo scene is generated procedurally in `diplab/demo.py`; no image assets
  are required.
