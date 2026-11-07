// Self-test: the notes' worked examples, ported from the Python selftest.
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "diplab/fft.hpp"
#include "diplab/filters.hpp"
#include "diplab/harris.hpp"
#include "diplab/histogram.hpp"
#include "diplab/hog.hpp"
#include "diplab/op.hpp"
#include "diplab/sharpening_common.hpp"

using namespace diplab;

namespace {

int g_pass = 0;
int g_fail = 0;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void nearly(double actual, double expected, double tolerance, const std::string& what) {
    if (!(std::fabs(actual - expected) <= tolerance))
        throw std::runtime_error(what + ": got " + std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
}

template <typename Fn>
void check(const std::string& name, Fn fn) {
    try {
        fn();
        ++g_pass;
        std::cout << "  PASS  " << name << "\n";
    } catch (const std::exception& e) {
        ++g_fail;
        std::cout << "  FAIL  " << name << ": " << e.what() << "\n";
    }
}

Image gray_from(const std::vector<std::vector<int>>& rows) {
    const int h = static_cast<int>(rows.size());
    const int w = static_cast<int>(rows[0].size());
    Image img(w, h, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img.at(y, x) = clamp_u8(rows[y][x]);
    return img;
}

Image random_rgb(int w, int h, unsigned seed) {
    Image img(w, h, 3);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (std::uint8_t& v : img.values) v = static_cast<std::uint8_t>(dist(rng));
    return img;
}

Image random_gray(int w, int h, unsigned seed) {
    Image img = random_rgb(w, h, seed);
    return to_gray_image(img);
}

Params P(std::initializer_list<std::pair<std::string, std::string>> items) {
    Params params;
    for (const auto& kv : items) params.set(kv.first, kv.second);
    return params;
}

Result run_op(const std::string& key, const Image& img, const Params& overrides = Params()) {
    const Operation* op = find_operation(key);
    if (op == nullptr) throw std::runtime_error("unknown operation: " + key);
    return apply(*op, img, overrides);
}

Plane plane_from(const std::vector<double>& values) {
    Plane p(static_cast<int>(values.size()), 1);
    for (std::size_t i = 0; i < values.size(); ++i) p.at(0, static_cast<int>(i)) = values[i];
    return p;
}

// ---------------------------------------------------------------- transforms
void test_fft_1d_example() {
    std::vector<Complex> f = {1, 2, 4, 4};
    fft_1d(f, false);
    nearly(f[0].real(), 11.0, 1e-9, "F(0)");
    nearly(f[0].imag(), 0.0, 1e-9, "F(0) imag");
    nearly(f[1].real(), -3.0, 1e-9, "F(1) real");
    nearly(f[1].imag(), 2.0, 1e-9, "F(1) imag");
    nearly(f[2].real(), -1.0, 1e-9, "F(2)");
    nearly(f[3].real(), -3.0, 1e-9, "F(3) real");
    nearly(f[3].imag(), -2.0, 1e-9, "F(3) imag");

    fft_1d(f, true);
    nearly(f[0].real(), 1.0, 1e-9, "inverse f0");
    nearly(f[1].real(), 2.0, 1e-9, "inverse f1");
    nearly(f[2].real(), 4.0, 1e-9, "inverse f2");
    nearly(f[3].real(), 4.0, 1e-9, "inverse f3");
}

void test_fft_roundtrip() {
    const int w = 32;
    const int h = 16;
    std::vector<Complex> a(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            a[static_cast<std::size_t>(y) * w + x] =
                std::sin(0.3 * x) + std::cos(0.11 * y) + 0.05 * x * y;
    const std::vector<Complex> original = a;
    fft_2d(a, w, h, false);
    fft_2d(a, w, h, true);
    for (std::size_t i = 0; i < a.size(); ++i)
        nearly(a[i].real(), original[i].real(), 1e-9, "roundtrip real");
}

void test_dft_dc_example() {
    // Note's 3x2 image: F(0,0) = sum = 60, F(0,0)/MN = 10.
    const Image img = gray_from({{5, 10, 20}, {5, 15, 5}});
    const int p = 2;  // rows padded to a power of two
    const int q = 4;  // columns padded to a power of two
    std::vector<Complex> f(static_cast<std::size_t>(p) * q, Complex(0.0, 0.0));
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) f[static_cast<std::size_t>(y) * q + x] = img.at(y, x);
    fft_2d(f, q, p, false);
    nearly(f[0].real(), 60.0, 1e-9, "F(0,0)");
    nearly(f[0].real() / static_cast<double>(img.width * img.height), 10.0, 1e-9, "F(0,0)/MN");
}

// -------------------------------------------------------------- enhancement
void test_grayscale_weights() {
    Image rgb(1, 1, 3);
    rgb.at(0, 0, 0) = 100;
    rgb.at(0, 0, 1) = 150;
    rgb.at(0, 0, 2) = 200;
    nearly(to_gray_plane(rgb).at(0, 0), 0.30 * 100 + 0.59 * 150 + 0.11 * 200, 1e-9,
           "weighted gray");
    nearly(to_gray_plane(rgb, true).at(0, 0), 150.0, 1e-9, "average gray");
}

void test_negative() {
    const Image img = gray_from({{0, 100, 255}});
    const Result r = run_op("negative", img);
    require(r.image.at(0, 0) == 255 && r.image.at(0, 1) == 155 && r.image.at(0, 2) == 0,
            "negative mapping");
}

void test_gamma() {
    Image ramp(16, 16, 1);
    for (int i = 0; i < 256; ++i) ramp.at(i / 16, i % 16) = static_cast<std::uint8_t>(i);
    const Result identity = run_op("gamma", ramp, P({{"gamma", "1"}, {"c", "1"}}));
    require(identity.image.values == ramp.values, "gamma 1 must be the identity");

    const Image flat = gray_from({{64, 64}, {64, 64}});
    require(run_op("gamma", flat, P({{"gamma", "0.5"}, {"c", "1"}})).image.at(0, 0) > 64,
            "gamma < 1 brightens");
    require(run_op("gamma", flat, P({{"gamma", "2"}, {"c", "1"}})).image.at(0, 0) < 64,
            "gamma > 1 darkens");
}

void test_log() {
    const Image img = gray_from({{0, 1, 50, 128, 255}});
    const Image out = run_op("log", img).image;
    require(out.at(0, 0) == 0, "log(0) = 0");
    require(out.at(0, 4) == 255, "log(max) = 255 with auto c");
    for (int x = 1; x < 5; ++x) require(out.at(0, x) >= out.at(0, x - 1), "log is monotonic");
}

void test_piecewise() {
    const Image img = gray_from({{0, 70, 180, 255}});
    const Image out = run_op("piecewise", img,
                             P({{"r1", "70"}, {"s1", "30"}, {"r2", "180"}, {"s2", "225"}}))
                          .image;
    require(out.at(0, 0) == 0 && out.at(0, 3) == 255, "end points");
    require(std::abs(static_cast<int>(out.at(0, 1)) - 30) <= 1, "first control point");
    require(std::abs(static_cast<int>(out.at(0, 2)) - 225) <= 1, "second control point");
}

void test_contrast_stretch() {
    const Image img = gray_from({{50, 60, 100, 180}});
    const Image out = run_op("contrast", img, P({{"low", "0"}, {"high", "100"}})).image;
    require(out.at(0, 0) == 0, "min -> 0");
    require(out.at(0, 3) == 255, "max -> 255");
    require(out.at(0, 1) == 20 || out.at(0, 1) == 19, "10/130*255 = 19.6");
}

void test_equalization_lut() {
    // 8-level worked example: 0->1, 1->3, 2->4, 3->5, 4->5, 5->6, 6->6, 7->7.
    const int counts[8] = {70, 100, 40, 60, 10, 70, 10, 40};
    std::vector<double> values;
    for (int level = 0; level < 8; ++level)
        for (int i = 0; i < counts[level]; ++i) values.push_back(level);
    const Plane channel = plane_from(values);
    const Lut lut = equalization_lut(channel, 8);
    const int expected[8] = {1, 3, 4, 5, 5, 6, 6, 7};
    for (int i = 0; i < 8; ++i)
        require(lut[static_cast<std::size_t>(i)] == expected[i],
                "lut[" + std::to_string(i) + "] = " + std::to_string(lut[i]));
}

// ---------------------------------------------------------------- spatial
const Image& note_image() {
    static const Image img = gray_from({{12, 16, 18, 35, 30, 28},
                                        {8, 50, 20, 24, 30, 22},
                                        {2, 12, 3, 21, 34, 18},
                                        {5, 6, 4, 3, 32, 12},
                                        {10, 1, 0, 4, 15, 23}});
    return img;
}

void test_box_example() {
    const Image out = run_op("box", note_image(), P({{"k", "3"}})).image;
    require(out.at(1, 1) == 16, "3x3 average around 50 gives 16, got " +
                                    std::to_string(out.at(1, 1)));
}

void test_weighted_example() {
    const Image out = run_op("weighted", note_image()).image;
    require(out.at(1, 1) == 22, "weighted average gives 22, got " +
                                    std::to_string(out.at(1, 1)));
}

void test_median_denoise() {
    Image base(64, 64, 1);
    std::fill(base.values.begin(), base.values.end(), 120);
    const Image noisy = run_op("salt", base, P({{"amount", "10"}, {"seed", "5"}})).image;
    const Image out = run_op("median", noisy, P({{"k", "3"}})).image;

    std::size_t bad = 0;
    for (std::uint8_t v : out.values)
        if (v != 120) ++bad;
    const double fraction = static_cast<double>(bad) / static_cast<double>(out.values.size());
    require(fraction < 0.01, "median leaves " + std::to_string(fraction) + " noisy pixels");
}

void test_order_stats() {
    const Image img = gray_from({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    require(run_op("min", img, P({{"k", "3"}})).image.at(1, 1) == 1, "min filter");
    require(run_op("max", img, P({{"k", "3"}})).image.at(1, 1) == 9, "max filter");
    require(run_op("nth", img, P({{"k", "3"}, {"rank", "5"}})).image.at(1, 1) == 5,
            "nth-order rank 5");
}

void test_adaptive_example() {
    const Image window = gray_from({{1, 3, 2}, {1, 11, 1}, {2, 4, 3}});
    const Image out =
        run_op("adaptive", window, P({{"noise_std", "1.41421356237"}, {"k", "3"}})).image;
    require(out.at(1, 1) == 9, "adaptive example gives 9, got " +
                                   std::to_string(out.at(1, 1)));
}

void test_adaptive_flat() {
    const Image noisy = random_gray(24, 24, 11);
    double in_sum = 0.0;
    for (std::uint8_t v : noisy.values) in_sum += v;
    const double in_mean = in_sum / static_cast<double>(noisy.values.size());

    const Image out = run_op("adaptive", noisy, P({{"noise_std", "3"}, {"k", "3"}})).image;
    double out_sum = 0.0;
    for (std::uint8_t v : out.values) out_sum += v;
    const double out_mean = out_sum / static_cast<double>(out.values.size());
    require(std::fabs(out_mean - in_mean) < 5.0, "adaptive keeps the mean");
}

void test_sobel_edge() {
    Image img(32, 32, 1);
    for (int y = 0; y < 32; ++y)
        for (int x = 16; x < 32; ++x) img.at(y, x) = 220;

    const Gradient g = gradients(img, "Sobel");
    std::size_t best = 0;
    for (std::size_t i = 1; i < g.magnitude.values.size(); ++i)
        if (g.magnitude.values[i] > g.magnitude.values[best]) best = i;
    const int column = static_cast<int>(best % 32);
    require(std::abs(column - 16) <= 2, "Sobel peak column " + std::to_string(column));
    require(g.magnitude.at(5, 5) < 1e-6, "Sobel is flat away from the edge");
}

void test_laplacian_constant() {
    Image img(16, 16, 1);
    std::fill(img.values.begin(), img.values.end(), 77);
    const Image out = run_op("laplacian", img).image;
    for (std::uint8_t v : out.values) require(v == 0, "Laplacian of a constant is 0");
}

void test_laplacian_sharpening_constant() {
    Image img(16, 16, 1);
    std::fill(img.values.begin(), img.values.end(), 90);
    const Image out = run_op("laplacian_sharp", img).image;
    for (std::uint8_t v : out.values) require(v == 90, "sharpening keeps a constant image");
}

void test_border_modes() {
    Image img = gray_from({{0, 1, 2, 3, 4},
                           {5, 6, 7, 8, 9},
                           {10, 11, 12, 13, 14},
                           {15, 16, 17, 18, 19},
                           {20, 21, 22, 23, 24}});
    for (const std::string border : {"reflect", "replicate", "zeros"}) {
        for (const std::string k : {"3", "5"}) {
            const Image out = run_op("box", img, P({{"k", k}, {"border", border}})).image;
            require(out.width == img.width && out.height == img.height,
                    "box " + border + " k=" + k + " keeps the size");
        }
    }
}

// -------------------------------------------------------------- frequency
void test_spectrum_dc() {
    Image img(16, 16, 1);
    std::fill(img.values.begin(), img.values.end(), 40);
    std::vector<Complex> f(256, Complex(0.0, 0.0));
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) f[static_cast<std::size_t>(y) * 16 + x] = img.at(y, x);
    fft_2d(f, 16, 16, false);
    fft_shift_2d(f, 16, 16, false);
    nearly(std::abs(f[8 * 16 + 8]), 40.0 * 256.0, 1e-6, "DC magnitude after fftshift");
}

void test_butterworth_half_power() {
    const int w = 32;
    const int h = 32;
    const Plane d = distance_grid(w, h);
    const Plane h_mask = lowpass_mask(w, h, 5.0, 2.0, MaskKind::Butterworth);
    int samples = 0;
    for (std::size_t i = 0; i < d.values.size(); ++i) {
        if (std::fabs(d.values[i] - 5.0) < 0.01) {
            nearly(h_mask.values[i], 0.5, 0.02, "Butterworth -3 dB point");
            ++samples;
        }
    }
    require(samples > 0, "no sample at D = D0");
}

void test_ideal_filters() {
    const int w = 32;
    const int h = 32;
    const Plane lp = lowpass_mask(w, h, 4.0, 2.0, MaskKind::Ideal);
    const Plane hp = highpass_mask(w, h, 4.0, 2.0, MaskKind::Ideal);
    require(lp.at(16, 16) == 1.0 && hp.at(16, 16) == 0.0, "centre values");
    require(lp.at(0, 0) == 0.0 && hp.at(0, 0) == 1.0, "corner values");
    for (std::size_t i = 0; i < lp.values.size(); ++i)
        nearly(lp.values[i] + hp.values[i], 1.0, 1e-12, "HP is the complement of LP");
}

void test_lowpass_preserves_brightness() {
    Image img(32, 32, 1);
    std::fill(img.values.begin(), img.values.end(), 100);
    const auto mask_fn = [](int w, int h) {
        return lowpass_mask(w, h, 6.0, 2.0, MaskKind::Gaussian);
    };
    const FilterOutput out = apply_frequency_filter(img, mask_fn, 1.0);
    require(std::fabs(out.image.mean() - 100.0) < 1.0, "low-pass keeps the mean");
}

void test_highpass_zero_mean() {
    Image img(32, 32, 1);
    std::fill(img.values.begin(), img.values.end(), 100);
    const auto mask_fn = [](int w, int h) {
        return highpass_mask(w, h, 6.0, 2.0, MaskKind::Gaussian);
    };
    const FilterOutput out = apply_frequency_filter(img, mask_fn, 1.0);
    require(std::fabs(out.image.mean()) < 1.0, "high-pass of a constant has mean 0");
}

void test_band_masks() {
    const int w = 32;
    const int h = 32;
    const Plane band = band_mask(w, h, 3.0, 8.0, 2.0, MaskKind::Ideal, false);
    const Plane low_hi = lowpass_mask(w, h, 8.0, 2.0, MaskKind::Ideal);
    const Plane low_lo = lowpass_mask(w, h, 3.0, 2.0, MaskKind::Ideal);
    const Plane reject = band_mask(w, h, 3.0, 8.0, 2.0, MaskKind::Ideal, true);
    for (std::size_t i = 0; i < band.values.size(); ++i) {
        const double manual = low_hi.values[i] * (1.0 - low_lo.values[i]);
        nearly(band.values[i], manual, 1e-12, "band-pass product");
        nearly(reject.values[i], 1.0 - manual, 1e-12, "band-reject complement");
    }
}

void test_notch_mask() {
    const int w = 32;
    const int h = 32;
    const Plane mask = notch_mask(w, h, 5.0, 3.0, 2.0);
    require(mask.at(16 + 3, 16 + 5) == 0.0, "notch at +du,+dv");
    require(mask.at(16 - 3, 16 - 5) == 0.0, "notch at -du,-dv");
    require(mask.at(16, 16) == 1.0, "DC survives");
}

void test_homomorphic() {
    const Image img = random_gray(48, 48, 6);
    const Image out =
        run_op("homomorphic", img,
               P({{"gamma_l", "0.5"}, {"gamma_h", "2"}, {"d0", "10"}, {"padding", "1"}}))
            .image;
    require(out.width == img.width && out.height == img.height, "homomorphic keeps the size");
    require(out.channels == 1, "homomorphic keeps the channel count");
}

// ----------------------------------------------------------------- features
void test_harris_square() {
    Image img(120, 120, 1);
    for (int y = 35; y < 85; ++y)
        for (int x = 35; x < 85; ++x) img.at(y, x) = 230;

    const auto points = harris_corners(img, 0.04, 3, 97.0, 6);
    const int expected[4][2] = {{35, 35}, {35, 84}, {84, 35}, {84, 84}};
    int hits = 0;
    for (const auto& corner : expected) {
        for (const auto& [py, px] : points) {
            if (std::abs(py - corner[0]) <= 6 && std::abs(px - corner[1]) <= 6) {
                ++hits;
                break;
            }
        }
    }
    require(hits >= 3, "Harris found " + std::to_string(hits) + " of 4 corners");
}

void test_harris_flat() {
    Image img(40, 40, 1);
    std::fill(img.values.begin(), img.values.end(), 100);
    const Plane response = harris_response_map(img);
    for (double v : response.values) nearly(v, 0.0, 1e-9, "flat Harris response");
}

void test_hog_length() {
    const Image img = random_gray(64, 128, 7);
    const HogDescriptor d = hog_compute(img, 8, 9, 2, 1, false);
    require(d.cells_y == 16 && d.cells_x == 8, "HOG cells 16x8");
    require(d.blocks_y == 15 && d.blocks_x == 7, "HOG blocks 15x7");
    require(d.vector.size() == 3780, "HOG length " + std::to_string(d.vector.size()));
}

void test_hog_normalisation() {
    const Image img = random_gray(64, 128, 8);
    const HogDescriptor d = hog_compute(img, 8, 9, 2, 1, false);
    const std::size_t block_size = 2 * 2 * 9;
    for (std::size_t start = 0; start + block_size <= d.vector.size(); start += block_size) {
        double norm2 = 0.0;
        for (std::size_t i = 0; i < block_size; ++i) norm2 += d.vector[start + i] * d.vector[start + i];
        nearly(std::sqrt(norm2), 1.0, 1e-3, "block L2 norm");
    }
}

void test_hog_bins() {
    Image img(16, 16, 1);
    for (int y = 0; y < 16; ++y)
        for (int x = 8; x < 16; ++x) img.at(y, x) = 100;
    const HogDescriptor d = hog_compute(img, 16, 9, 1, 1, false);
    require(d.cells_y == 1 && d.cells_x == 1, "one cell");
    std::size_t best = 0;
    double best_value = -1.0;
    for (int b = 0; b < 9; ++b) {
        if (d.hist[static_cast<std::size_t>(b)] > best_value) {
            best_value = d.hist[static_cast<std::size_t>(b)];
            best = static_cast<std::size_t>(b);
        }
    }
    require(best_value > 0.0, "the edge produces votes");
    require(best == 0, "vertical edge goes to bin 0, got " + std::to_string(best));
}

// ------------------------------------------------------------------- utils
void test_resize_shape() {
    Plane src(30, 20);
    for (int y = 0; y < 20; ++y)
        for (int x = 0; x < 30; ++x) src.at(y, x) = x + y;
    const Plane out = resize_bilinear(src, 15, 10);
    require(out.width == 15 && out.height == 10, "resize returns the requested shape");
}

void test_all_operations_run() {
    const Image rgb = random_rgb(96, 96, 9);
    const Image gray = to_gray_image(rgb);
    std::vector<std::string> failures;
    for (const Operation& op : operations()) {
        for (const auto& [label, img] : {std::pair<const char*, const Image*>{"rgb", &rgb},
                                         std::pair<const char*, const Image*>{"gray", &gray}}) {
            try {
                const Result r = apply(op, *img, Params());
                if (r.image.empty()) failures.push_back(op.key + "/" + label + ": empty");
                if (r.image.channels != 1 && r.image.channels != 3)
                    failures.push_back(op.key + "/" + label + ": channels " +
                                       std::to_string(r.image.channels));
                if (r.image.values.size() !=
                    static_cast<std::size_t>(r.image.width) * r.image.height * r.image.channels)
                    failures.push_back(op.key + "/" + label + ": buffer size mismatch");
                for (const auto& aux : r.aux)
                    if (aux.second.empty())
                        failures.push_back(op.key + "/" + label + " aux " + aux.first);
            } catch (const std::exception& e) {
                failures.push_back(op.key + "/" + label + ": " + e.what());
            }
        }
    }
    std::string message;
    for (const std::string& f : failures) message += "\n  " + f;
    require(failures.empty(), "operations failed:" + message);
    require(operations().size() == 32,
            "expected 32 operations, got " + std::to_string(operations().size()));
}

}  // namespace

int main() {
    std::cout << "FFT / DFT\n";
    check("FFT of [1,2,4,4] -> [11, -3+2j, -1, -3-2j]", test_fft_1d_example);
    check("FFT round trip on 2-D data", test_fft_roundtrip);
    check("2-D example: F(0,0)/MN = 10", test_dft_dc_example);

    std::cout << "\nEnhancement\n";
    check("weighted grayscale uses 0.3 / 0.59 / 0.11", test_grayscale_weights);
    check("negative: s = 255 - r", test_negative);
    check("gamma = 1 is the identity; <1 bright, >1 dark", test_gamma);
    check("log: 0 -> 0, max -> 255, monotonic", test_log);
    check("piecewise linear hits its control points", test_piecewise);
    check("contrast stretch maps min -> 0, max -> 255", test_contrast_stretch);
    check("equalization matches the 8-level example", test_equalization_lut);

    std::cout << "\nSpatial filtering\n";
    check("3x3 average around 50 gives 16", test_box_example);
    check("weighted 1/16 average gives 22", test_weighted_example);
    check("median removes salt-and-pepper noise", test_median_denoise);
    check("min / max / nth-order are consistent", test_order_stats);
    check("adaptive example: 11 -> 9", test_adaptive_example);
    check("adaptive keeps the local mean", test_adaptive_flat);
    check("Sobel responds to a vertical edge only", test_sobel_edge);
    check("Laplacian of a constant is zero", test_laplacian_constant);
    check("Laplacian sharpening keeps a constant image", test_laplacian_sharpening_constant);
    check("border modes keep the output size", test_border_modes);

    std::cout << "\nFrequency domain\n";
    check("spectrum: DC at the centre after fftshift", test_spectrum_dc);
    check("Butterworth is 0.5 at D = D0", test_butterworth_half_power);
    check("ideal LP/HP centre, corner and complement", test_ideal_filters);
    check("low-pass on a constant keeps brightness", test_lowpass_preserves_brightness);
    check("high-pass of a constant has mean 0", test_highpass_zero_mean);
    check("band masks are LP * HP; reject complements", test_band_masks);
    check("notch mask zeroes both symmetric spikes", test_notch_mask);
    check("homomorphic keeps size and channels", test_homomorphic);

    std::cout << "\nFeatures\n";
    check("Harris finds the corners of a bright square", test_harris_square);
    check("Harris response is 0 on a flat image", test_harris_flat);
    check("HOG length is 3780 for 64x128 / cell 8 / 2x2", test_hog_length);
    check("HOG blocks are L2-normalised", test_hog_normalisation);
    check("HOG bins a vertical edge into bin 0", test_hog_bins);

    std::cout << "\nUtils + integration\n";
    check("resize returns the requested shape", test_resize_shape);
    check("every operation runs on colour and grayscale", test_all_operations_run);

    std::cout << "\n" << std::string(60, '=') << "\n"
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
