"""DIP Lab — one topic folder per lecture section, one algorithm per file.

Layout
------
core        : shared data model, dtype/colour/border helpers, FFT helpers,
              and the naive DFT reference used by the worked example
foundations : grayscale conversion                             (notes 00)
point_processing : negative, log, gamma, piecewise, stretch, equalize
noise       : salt & pepper and Gaussian noise generators
smoothing   : box, weighted, median, min, max, nth-order, adaptive (notes 01)
sharpening  : Sobel/Scharr, Laplacian, Laplacian sharpening       (notes 01)
frequency   : spectrum, low/high/band/notch, homomorphic          (notes 02)
features    : Harris corners and HOG                              (notes 03)
operations  : registry that collects every module's OPERATION
demo        : synthetic test scene
gui         : PyQt5 browser, preview, parameter panel and widgets
"""

__version__ = "2.0.0"
