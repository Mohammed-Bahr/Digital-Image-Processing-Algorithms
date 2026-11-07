// Histogram of Oriented Gradients: unsigned gradients, 9 bins, L2 blocks.
#include "diplab/hog.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "diplab/filters.hpp"
#include "diplab/fft.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

void poke(Image& img, int y, int x, std::uint8_t value) {
    if (y < 0 || y >= img.height || x < 0 || x >= img.width) return;
    if (value > img.at(y, x, 0)) img.at(y, x, 0) = value;
}

}  // namespace

HogDescriptor hog_compute(const Image& img, int cell, int bins, int block, int stride,
                          bool resize_to_window) {
    cell = std::max(cell, 1);
    bins = std::max(bins, 1);
    block = std::max(block, 1);
    stride = std::max(stride, 1);

    Plane gray = to_gray_plane(img);
    if (resize_to_window) gray = resize_bilinear(gray, 64, 128);

    const Kernel grad_x = {{-1.0, 0.0, 1.0}};
    const Kernel grad_y = {{-1.0}, {0.0}, {1.0}};
    const Plane gx = correlate(gray, grad_x, Border::Reflect);
    const Plane gy = correlate(gray, grad_y, Border::Reflect);

    Plane magnitude(gray.width, gray.height);
    Plane angle(gray.width, gray.height);
    for (std::size_t i = 0; i < gray.values.size(); ++i) {
        magnitude.values[i] = std::hypot(gx.values[i], gy.values[i]);
        double a = std::atan2(gy.values[i], gx.values[i]);
        a -= kPi * std::floor(a / kPi);  // fold to [0, pi): unsigned orientation
        angle.values[i] = a;
    }

    const int ny = gray.height / cell;
    const int nx = gray.width / cell;
    if (ny < block || nx < block)
        throw std::invalid_argument("image is too small for the selected cell/block size");

    std::vector<double> hist(static_cast<std::size_t>(ny) * nx * bins, 0.0);
    const double bin_scale = bins / kPi;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double* cell_hist = &hist[(static_cast<std::size_t>(j) * nx + i) * bins];
            for (int y = j * cell; y < (j + 1) * cell; ++y) {
                for (int x = i * cell; x < (i + 1) * cell; ++x) {
                    const double pos = angle.at(y, x) * bin_scale;
                    const double base = std::floor(pos);
                    const int lower = static_cast<int>(std::fmod(base, bins)) % bins;
                    const int upper = (lower + 1) % bins;
                    const double frac = pos - base;
                    const double m = magnitude.at(y, x);
                    cell_hist[lower] += m * (1.0 - frac);
                    cell_hist[upper] += m * frac;
                }
            }
        }
    }

    const int by = (ny - block) / stride + 1;
    const int bx = (nx - block) / stride + 1;
    HogDescriptor d;
    d.hist = hist;
    d.cells_y = ny;
    d.cells_x = nx;
    d.blocks_y = by;
    d.blocks_x = bx;
    d.cell = cell;
    d.bins = bins;
    d.block = block;
    d.width = gray.width;
    d.height = gray.height;
    d.magnitude = magnitude;
    d.angle = angle;
    d.vector.reserve(static_cast<std::size_t>(by) * bx * block * block * bins);

    for (int j = 0; j < by; ++j) {
        for (int i = 0; i < bx; ++i) {
            std::vector<double> part;
            part.reserve(static_cast<std::size_t>(block) * block * bins);
            for (int cj = 0; cj < block; ++cj) {
                for (int ci = 0; ci < block; ++ci) {
                    const int cy = j * stride + cj;
                    const int cx = i * stride + ci;
                    const double* cell_hist = &hist[(static_cast<std::size_t>(cy) * nx + cx) * bins];
                    part.insert(part.end(), cell_hist, cell_hist + bins);
                }
            }
            double norm2 = 1e-6;
            for (double v : part) norm2 += v * v;
            const double norm = std::sqrt(norm2);
            for (double v : part) d.vector.push_back(v / norm);
        }
    }
    return d;
}

Image hog_visualization(const std::vector<double>& hist, int cells_y, int cells_x,
                        int bins, int scale) {
    Image out(cells_x * scale, cells_y * scale, 1);
    std::fill(out.values.begin(), out.values.end(), 22);
    if (hist.empty()) return out;

    double peak = 0.0;
    for (double v : hist) peak = std::max(peak, v);
    if (peak <= 0.0) return out;

    for (int j = 0; j < cells_y; ++j) {
        for (int i = 0; i < cells_x; ++i) {
            const double* cell_hist = &hist[(static_cast<std::size_t>(j) * cells_x + i) * bins];
            double cmax = 0.0;
            for (int b = 0; b < bins; ++b) cmax = std::max(cmax, cell_hist[b]);
            if (cmax <= 0.0) continue;

            const double cy = j * scale + scale / 2.0;
            const double cx = i * scale + scale / 2.0;
            for (int b = 0; b < bins; ++b) {
                const double ratio = std::min(cell_hist[b] / cmax, 1.0);
                if (ratio <= 0.0) continue;
                const double angle = (b + 0.5) * kPi / bins;
                const double length = ratio * (scale * 0.5 - 1.0);
                const double dy = std::sin(angle) * length;
                const double dx = std::cos(angle) * length;

                const int n = std::max(static_cast<int>(2 * length) + 3, 3);
                const auto value = static_cast<std::uint8_t>(std::min(90.0 + 165.0 * ratio, 255.0));
                for (int k = 0; k < n; ++k) {
                    const double t = (n > 1) ? -1.0 + 2.0 * k / (n - 1) : 0.0;
                    int y = static_cast<int>(std::lround(cy + t * dy));
                    int x = static_cast<int>(std::lround(cx + t * dx));
                    y = std::min(std::max(y, j * scale), (j + 1) * scale - 1);
                    x = std::min(std::max(x, i * scale), (i + 1) * scale - 1);
                    poke(out, y, x, value);
                    poke(out, y + 1, x, value);
                    poke(out, y, x + 1, value);
                }
            }
            poke(out, static_cast<int>(cy), static_cast<int>(cx), 255);
        }
    }

    for (int y = 0; y < out.height; y += scale)
        for (int x = 0; x < out.width; ++x) poke(out, y, x, 42);
    for (int x = 0; x < out.width; x += scale)
        for (int y = 0; y < out.height; ++y) poke(out, y, x, 42);
    return out;
}

namespace {

Result run(const Image& img, const Params& params) {
    const int cell = params.integer("cell", 8);
    const int bins = params.integer("bins", 9);
    const int block = params.integer("block", 2);
    const int stride = params.integer("stride", 1);
    const bool resize_to_window = params.text("resize", "64 x 128").rfind("64", 0) == 0;

    const HogDescriptor d = hog_compute(img, cell, bins, block, stride, resize_to_window);
    const int scale = std::min(std::max(440 / std::max(d.cells_x, 1), 6), 26);
    const std::size_t length =
        static_cast<std::size_t>(d.blocks_y) * d.blocks_x * block * block * bins;

    Plane magnitude = d.magnitude;
    normalize_minmax(magnitude, 0.0, std::max(magnitude.maximum(), 1e-12));

    Result r;
    r.image = hog_visualization(d.hist, d.cells_y, d.cells_x, bins, scale);
    r.aux.push_back({"Gradient magnitude", plane_to_image(magnitude)});
    r.meta["length"] = static_cast<double>(length);
    r.log = "window " + std::to_string(d.width) + "x" + std::to_string(d.height) +
            " px, cell " + std::to_string(cell) + "x" + std::to_string(cell) + ", " +
            std::to_string(bins) + " bins, block " + std::to_string(block) + "x" +
            std::to_string(block) + " cells, stride " + std::to_string(stride) +
            "\ncells: " + std::to_string(d.cells_x) + " x " + std::to_string(d.cells_y) +
            ", block positions: " + std::to_string(d.blocks_x) + " x " +
            std::to_string(d.blocks_y) + "\ndescriptor length = " +
            std::to_string(static_cast<long long>(length)) +
            ".\n9 bins = 20° each; every pixel votes softly for the two nearest bins;\n"
            "each block is L2-normalised to cancel contrast changes.";
    return r;
}

}  // namespace

Operation op_hog() {
    Operation op;
    op.key = "hog";
    op.name = "HOG descriptor";
    op.category = "Features";
    op.description = "Gradients -> cell histograms -> overlapping L2-normalised blocks.";
    op.run = run;
    Param resize;
    resize.key = "resize";
    resize.label = "Resize first";
    resize.kind = "choice";
    resize.def = "64 x 128";
    resize.choices = {"64 x 128", "keep size"};
    Param cell;
    cell.key = "cell";
    cell.label = "Cell size";
    cell.kind = "int";
    cell.def = "8";
    cell.minimum = 4;
    cell.maximum = 32;
    cell.step = 4;
    Param bins;
    bins.key = "bins";
    bins.label = "Orientation bins";
    bins.kind = "int";
    bins.def = "9";
    bins.minimum = 4;
    bins.maximum = 18;
    Param block;
    block.key = "block";
    block.label = "Block (cells)";
    block.kind = "int";
    block.def = "2";
    block.minimum = 1;
    block.maximum = 4;
    Param stride;
    stride.key = "stride";
    stride.label = "Block stride";
    stride.kind = "int";
    stride.def = "1";
    stride.minimum = 1;
    stride.maximum = 3;
    op.params = {resize, cell, bins, block, stride};
    return op;
}

}  // namespace diplab
