#include "diplab/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
using namespace std;

namespace diplab {

int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return max(p, 1);
}

void fft_1d(vector<Complex>& a, bool inverse) {
    const size_t n = a.size();
    if (n <= 1) return;
    if (n & (n - 1)) throw invalid_argument("fft length must be a power of two");

    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const Complex wlen(cos(ang), sin(ang));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
        for (Complex& x : a) x *= scale;
    }
}

void fft_2d(vector<Complex>& a, int width, int height, bool inverse) {
    if (width <= 0 || height <= 0) return;
    vector<Complex> row(static_cast<size_t>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) row[x] = a[static_cast<size_t>(y) * width + x];
        fft_1d(row, inverse);
        for (int x = 0; x < width; ++x) a[static_cast<size_t>(y) * width + x] = row[x];
    }
    vector<Complex> col(static_cast<size_t>(height));
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) col[y] = a[static_cast<size_t>(y) * width + x];
        fft_1d(col, inverse);
        for (int y = 0; y < height; ++y) a[static_cast<size_t>(y) * width + x] = col[y];
    }
}

void fft_shift_2d(vector<Complex>& a, int width, int height, bool inverse_shift) {
    if (width <= 0 || height <= 0) return;
    vector<Complex> out(a.size());
    const int oy = inverse_shift ? (height + 1) / 2 : height / 2;
    const int ox = inverse_shift ? (width + 1) / 2 : width / 2;
    for (int y = 0; y < height; ++y) {
        const int dy = ((y + oy) % height + height) % height;
        for (int x = 0; x < width; ++x) {
            const int dx = ((x + ox) % width + width) % width;
            out[static_cast<size_t>(dy) * width + dx] =
                a[static_cast<size_t>(y) * width + x];
        }
    }
    a.swap(out);
}

Plane distance_grid(int width, int height) {
    Plane d(width, height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            d.at(y, x) = hypot(static_cast<double>(y - height / 2),
                                    static_cast<double>(x - width / 2));
    return d;
}

MaskKind parse_mask_kind(const string& name) {
    if (name == "ideal") return MaskKind::Ideal;
    if (name == "gaussian") return MaskKind::Gaussian;
    return MaskKind::Butterworth;
}

bool mask_kind_valid(const string& name) {
    return name == "ideal" || name == "butterworth" || name == "gaussian";
}

Plane lowpass_mask(int width, int height, double d0, double order, MaskKind kind) {
    const Plane d = distance_grid(width, height);
    const double cutoff = max(d0, kEps);
    Plane h(width, height);
    for (size_t i = 0; i < d.values.size(); ++i) {
        const double dist = d.values[i];
        switch (kind) {
            case MaskKind::Ideal:
                h.values[i] = (dist <= cutoff) ? 1.0 : 0.0;
                break;
            case MaskKind::Gaussian:
                h.values[i] = exp(-(dist * dist) / (2.0 * cutoff * cutoff));
                break;
            case MaskKind::Butterworth:
            default:
                h.values[i] = 1.0 / (1.0 + pow(dist / cutoff, 2.0 * max(order, 0.01)));
                break;
        }
    }
    return h;
}

Plane highpass_mask(int width, int height, double d0, double order, MaskKind kind) {
    Plane h = lowpass_mask(width, height, d0, order, kind);
    for (double& v : h.values) v = 1.0 - v;
    return h;
}

Plane band_mask(int width, int height, double d0, double d1, double order,
                MaskKind kind, bool reject) {
    const double lo = max(min(d0, d1), kEps);
    const double hi = max(max(d0, d1), kEps);
    const Plane low_hi = lowpass_mask(width, height, hi, order, kind);
    const Plane low_lo = lowpass_mask(width, height, lo, order, kind);
    Plane h(width, height);
    for (size_t i = 0; i < h.values.size(); ++i) {
        const double bp = low_hi.values[i] * (1.0 - low_lo.values[i]);
        h.values[i] = reject ? 1.0 - bp : bp;
    }
    return h;
}

Plane notch_mask(int width, int height, double du, double dv, double radius) {
    const double r = max(radius, kEps);
    Plane h(width, height, 1.0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double cy = static_cast<double>(y - height / 2);
            const double cx = static_cast<double>(x - width / 2);
            const double d1 = hypot(cy - dv, cx - du);
            const double d2 = hypot(cy + dv, cx + du);
            if (d1 <= r || d2 <= r) h.at(y, x) = 0.0;
        }
    }
    return h;
}

pair<int, int> padded_size(int height, int width, double factor) {
    const double f = max(1.0, factor);
    const int p = next_power_of_two(static_cast<int>(ceil(height * f)));
    const int q = next_power_of_two(static_cast<int>(ceil(width * f)));
    return {max(p, 8), max(q, 8)};
}

Plane spectrum_display(const vector<Complex>& centred, int width, int height,
                       double gamma) {
    Plane mag(width, height);
    double peak = 0.0;
    for (size_t i = 0; i < mag.values.size(); ++i) {
        mag.values[i] = log1p(abs(centred[i]));
        peak = max(peak, mag.values[i]);
    }
    Plane out(width, height);
    if (peak <= 0.0) return out;
    const double g = max(gamma, kEps);
    for (size_t i = 0; i < out.values.size(); ++i)
        out.values[i] = pow(mag.values[i] / peak, g) * 255.0;
    return out;
}

Plane image_spectrum(const Image& img, double pad_factor, int& padded_w, int& padded_h) {
    const Plane gray = to_gray_plane(img);
    const auto size = padded_size(gray.height, gray.width, pad_factor);
    const int p = size.first;
    const int q = size.second;
    vector<Complex> f(static_cast<size_t>(p) * q, Complex(0.0, 0.0));
    for (int y = 0; y < gray.height; ++y)
        for (int x = 0; x < gray.width; ++x)
            f[static_cast<size_t>(y) * q + x] = gray.at(y, x);
    fft_2d(f, q, p, false);
    fft_shift_2d(f, q, p, false);
    padded_w = q;
    padded_h = p;
    return spectrum_display(f, q, p);
}

FilterOutput apply_frequency_filter(const Image& img,
                                    const function<Plane(int, int)>& mask_fn,
                                    double pad_factor) {
    FilterOutput out;
    const Plane gray = to_gray_plane(img);
    const auto size = padded_size(img.height, img.width, pad_factor);
    const int p = size.first;  // rows
    const int q = size.second; // columns

    const Plane mask = mask_fn(q, p);
    if (mask.width != q || mask.height != p)
        throw invalid_argument("filter mask has the wrong size");

    auto filter_plane = [&](const Plane& ch) {
        vector<Complex> f(static_cast<size_t>(p) * q, Complex(0.0, 0.0));
        for (int y = 0; y < ch.height; ++y)
            for (int x = 0; x < ch.width; ++x) f[static_cast<size_t>(y) * q + x] = ch.at(y, x);
        fft_2d(f, q, p, false);
        fft_shift_2d(f, q, p, false);
        for (size_t i = 0; i < f.size(); ++i) f[i] *= mask.values[i];
        fft_shift_2d(f, q, p, true);
        fft_2d(f, q, p, true);
        Plane o(ch.width, ch.height);
        for (int y = 0; y < ch.height; ++y)
            for (int x = 0; x < ch.width; ++x)
                o.at(y, x) = f[static_cast<size_t>(y) * q + x].real();
        return o;
    };

    out.image = map_channels(img, filter_plane);
    out.mask = mask;
    out.padded_width = q;
    out.padded_height = p;

    vector<Complex> f_in(static_cast<size_t>(p) * q, Complex(0.0, 0.0));
    for (int y = 0; y < gray.height; ++y)
        for (int x = 0; x < gray.width; ++x)
            f_in[static_cast<size_t>(y) * q + x] = gray.at(y, x);
    fft_2d(f_in, q, p, false);
    fft_shift_2d(f_in, q, p, false);
    out.spectrum_input = spectrum_display(f_in, q, p);
    vector<Complex> f_out = f_in;
    for (size_t i = 0; i < f_out.size(); ++i) f_out[i] *= mask.values[i];
    out.spectrum_output = spectrum_display(f_out, q, p);
    return out;
}

}  // namespace diplab
