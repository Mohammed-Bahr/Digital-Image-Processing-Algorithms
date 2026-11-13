# DIP Lab — C++ port (pure math, one algorithm per file)

A dependency-free C++17 implementation of the algorithms in the Python DIP Lab.
No NumPy, no OpenCV, no SciPy: the FFT, correlation, order-statistic filters,
Harris response and HOG descriptor are written from the formulas in the notes.

```
cpp/
  include/diplab/          public headers (image, filters, fft, op, ...)
  src/core/                shared, algorithm-independent code
  src/<topic>/             one .cpp per algorithm
  src/cli.cpp              the `diplab` command line tool
  gui/                     raylib front end (optional)
  tests/selftest.cpp       the notes' worked examples as automatic checks
```

## Build

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Produces `build/diplab` (CLI) and `build/diplab_selftest` (checks), plus
`build/diplab_gui` when raylib is installed. The core and the checks need only a
C++17 compiler.

```bash
./build/diplab_selftest      # or: ctest --test-dir build
```

Current result: **35 checks, all passing**.

## Use

```bash
./build/diplab demo scene.ppm                   # synthetic test scene
./build/diplab list                             # all algorithms
./build/diplab list --op lowpass                # its parameters

./build/diplab run box scene.ppm out.ppm k=5 border=reflect
./build/diplab run lowpass scene.ppm out.ppm kind=gaussian d0=25 --log
./build/diplab run gamma scene.ppm out.ppm gamma=0.6
./build/diplab run harris scene.ppm corners.bmp --save-aux
```

Parameters are plain `key=value` pairs (a leading `--` is accepted) and come
from each operation's declaration, so `list --op KEY` is the reference.
`--log` prints the formula/explanation; `--save-aux` also writes the auxiliary
images (spectra, filter mask, Harris response, gradient magnitude, ...).

Formats: `.pgm` / `.ppm` (binary PGM/PPM) and 24-bit `.bmp`, read and write.
Convert other formats with ImageMagick (`magick photo.jpg photo.ppm`).

## GUI (raylib)

The same library is wrapped in a raylib desktop window. The algorithms are not
reimplemented or modified for it: the GUI reads the operation registry, draws
the parameters each operation declares and calls the same `apply`.

Requirements: raylib 5.x visible to `pkg-config` (`pkg-config --libs raylib`),
for example `sudo apt install libraylib-dev`. When raylib is missing the GUI
target is skipped and the rest still builds; `-DDIPLAB_BUILD_GUI=OFF` disables
it explicitly.

```bash
./build/diplab_gui                        # opens on the box filter
./build/diplab_gui --op lowpass           # start on another algorithm
./build/diplab_gui --image photo.jpg      # start on your own image
./build/diplab_gui --gallery              # start in the gallery
./build/diplab_gui --screenshot 2 --out preview.png   # save a frame and exit
```

What it shows:

* **Viewport** — one large canvas with a draggable split: input on the left,
  output on the right (or Input / Output only).
* **Histogram** — input and output luminance histograms overlaid, with means and
  standard deviations, so the effect of each operation is visible at a glance.
* **Sidebar** — all 32 algorithms grouped by topic, with counts and a search box.
* **Gallery** — every algorithm applied to the current input at default settings,
  with timings; click a card to open its parameters.
* **Inspector** — description, live parameter controls, Apply / Reset / Save and
  the operation's formatted explanation (its spectrum mask mean, corner count, ...).

Controls: `F5` re-run, `S` save the result as `diplab_<key>.ppm`, `G` toggle the
gallery, `R` reset parameters, `Esc` leave the search box or the gallery, mouse
wheel scrolls the sidebar, explanation and gallery. Drag any PNG, JPEG, BMP,
PGM or PPM file onto the window to load it, or pass `--image PATH`; large images
are scaled down to 1024 px on the long side for the working copy, and the
viewport and gallery follow the new input.

Only the GUI decodes PNG/JPEG (through raylib's loader); the library and CLI
stay dependency-free, so for the CLI convert first, e.g.
`convert photo.jpg photo.ppm && ./build/diplab run gamma photo.ppm out.ppm`.

`--screenshot SEC --out PATH` writes one PNG after `SEC` seconds and quits, which
is how the layout was checked without a compositor. Note that raylib writes the
PNG to the working directory using only the file name.

## Layout and algorithm map

| File | Formula / idea |
|---|---|
| `foundations/grayscale_weighted.cpp` | `I = 0.30 R + 0.59 G + 0.11 B` |
| `foundations/grayscale_average.cpp` | `I = (R + G + B) / 3` |
| `foundations/extract_channel.cpp` | one colour plane |
| `point_processing/negative.cpp` | `s = 255 - r` |
| `point_processing/logarithmic.cpp` | `s = c log(1 + r)` |
| `point_processing/gamma.cpp` | `s = c r^γ`, `r` in 0..1 |
| `point_processing/piecewise_linear.cpp` | three segments through `(r1,s1)`, `(r2,s2)` |
| `point_processing/contrast_stretching.cpp` | `(X − Xmin)/(Xmax − Xmin)·255`, percentiles |
| `point_processing/histogram_equalization.cpp` | `s_k = round((L−1)·CDF(k))` |
| `noise/salt_and_pepper.cpp` | random pixels to 0 or 255 |
| `noise/gaussian_noise.cpp` | `g = f + η`, `η ~ N(0, σ²)` |
| `smoothing/box_filter.cpp` | uniform `k×k` mean |
| `smoothing/weighted_average.cpp` | `(1/16)·[[1,2,1],[2,4,2],[1,2,1]]` |
| `smoothing/median_filter.cpp` | middle rank |
| `smoothing/min_filter.cpp` / `max_filter.cpp` | rank 1 / rank `k²` |
| `smoothing/nth_order.cpp` | arbitrary rank |
| `smoothing/adaptive_local.cpp` | `f̂ = g − (σ_η²/σ_L²)(g − m_L)` |
| `sharpening/sobel.cpp` | `G = √(Gx² + Gy²)` (Sobel or Scharr) |
| `sharpening/sobel_orientation.cpp` | `θ = atan2(Gy, Gx)` as hue |
| `sharpening/laplacian.cpp` | 4-neighbour / 8-neighbour / diagonal masks |
| `sharpening/laplacian_sharpening.cpp` | `g = f − c ∇²f` |
| `frequency/spectrum.cpp` | centred `log(1 + |F|)` |
| `frequency/lowpass.cpp` | ideal / Butterworth / Gaussian |
| `frequency/highpass.cpp` | complement `1 − H_low` |
| `frequency/bandpass.cpp` | `LP(D1) · HP(D0)` |
| `frequency/bandreject.cpp` | complement of the band-pass |
| `frequency/notch.cpp` | two symmetric discs zeroed |
| `frequency/homomorphic.cpp` | `log → DFT → H → IDFT → exp` |
| `features/harris.cpp` | `R = det(M) − k·trace(M)²` + markers |
| `features/harris_response.cpp` | positive part of `R` |
| `features/hog.cpp` | 9-bin soft binning, L2 blocks, star picture |

Shared code lives in `src/core/`:

* `image.cpp` — `Plane` / `Image`, grayscale weights, LUTs, bilinear resize
* `filters.cpp` — border handling (`reflect`/`replicate`/`zeros`), correlation,
  separable box mean, order statistics, percentiles
* `fft.cpp` — iterative radix-2 FFT, fftshift, distance grid, all frequency
  masks and the `f → DFT → H → IDFT → g` pipeline
* `io.cpp` — PGM/PPM and BMP
* `op.cpp` — parameters and the registry that collects the 32 operations
* `demo.cpp` — the procedural 512×512 scene

## Notes on the math

* **Correlation, not convolution**: `g(x,y) = Σ w(s,t)·f(x+s,y+t)`, matching the
  notes' convention (OpenCV's `filter2D` without a flipped kernel).
* **Border**: `reflect` is symmetric padding (`fedcba|abcdefgh|hgfedcb`),
  `replicate` clamps, `zeros` pads with 0.
* **Equalization** deliberately omits `cdf_min`, reproducing the worked example
  `0→1, 1→3, 2→4, 3→5, 4→5, 5→6, 6→6, 7→7`.
* **FFT**: power-of-two, in-place Cooley–Tukey; images are zero-padded to the
  next power of two and cropped back, the mask is applied on the centred
  spectrum, so DC lines up with the mask centre.
* **Harris**: Sobel derivatives, box aggregation, `R = det − k·trace²`, 3×3
  non-maximum suppression, percentile threshold and minimum-distance thinning.

## Differences from the Python version

* **GUI toolkit.** The front end is raylib (OpenGL) instead of PyQt5. It exposes
  the same 32 algorithms, parameters and explanations, but draws its own widgets
  and adds a split compare view, an overlaid histogram and a gallery grid.
* **Own numerics.** The Python code calls OpenCV/SciPy; here every filter is an
  explicit loop, and the FFT is hand-written, so results can differ in the last
  bit and in interpolation details of filter tails.
* **Noise generators** use `std::mt19937`, so a given seed does not reproduce
  NumPy's exact noise pattern.
* **Image IO.** The library and CLI read/write PGM/PPM/BMP only instead of
  PIL's full format list; the GUI additionally opens anything raylib decodes
  (PNG, JPEG, GIF, TGA, BMP) and converts it to the internal image.

## Adding an algorithm

1. Create `src/<topic>/my_algo.cpp` with a `Result run(const Image&, const Params&)`
   and an `Operation op_my_algo()` factory.
2. Declare the factory in `include/diplab/operations.hpp`.
3. Add `op_my_algo()` to the list in `src/core/op.cpp`.

`list`, `run`, `--log` and the self-test pick it up automatically.
