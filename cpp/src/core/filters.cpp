#include "diplab/filters.hpp"

#include <algorithm>
#include <cmath>

namespace diplab {

Border parse_border(const std::string& name) {
    if (name == "replicate" || name == "edge") return Border::Replicate;
    if (name == "zeros" || name == "constant") return Border::Zeros;
    return Border::Reflect;  // symmetric padding, as in the notes
}

int border_index(int i, int n, Border mode) {
    if (n <= 1) return 0;
    switch (mode) {
        case Border::Replicate:
            if (i < 0) return 0;
            if (i >= n) return n - 1;
            return i;
        case Border::Zeros:
            return (i < 0 || i >= n) ? -1 : i;
        case Border::Reflect:
        default: {
            // fedcba | abcdefgh | hgfedcb
            while (i < 0 || i >= n) {
                if (i < 0) i = -i - 1;
                if (i >= n) i = 2 * n - i - 1;
            }
            return i;
        }
    }
}

Plane correlate(const Plane& src, const Kernel& kernel, Border border) {
    if (src.empty() || kernel.empty()) return Plane(src.width, src.height);
    const int kh = static_cast<int>(kernel.size());
    const int kw = static_cast<int>(kernel[0].size());
    const int cy = kh / 2;
    const int cx = kw / 2;
    Plane out(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double acc = 0.0;
            for (int j = 0; j < kh; ++j) {
                const int sy = border_index(y + j - cy, src.height, border);
                if (sy < 0) continue;
                for (int i = 0; i < kw; ++i) {
                    const int sx = border_index(x + i - cx, src.width, border);
                    if (sx < 0) continue;
                    acc += kernel[j][i] * src.at(sy, sx);
                }
            }
            out.at(y, x) = acc;
        }
    }
    return out;
}

Plane box_mean(const Plane& src, int k, Border border) {
    if (k < 1) k = 1;
    const int half = k / 2;
    Plane tmp(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double acc = 0.0;
            for (int i = 0; i < k; ++i) {
                const int sx = border_index(x + i - half, src.width, border);
                acc += (sx < 0) ? 0.0 : src.at(y, sx);
            }
            tmp.at(y, x) = acc / k;
        }
    }
    Plane out(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            double acc = 0.0;
            for (int j = 0; j < k; ++j) {
                const int sy = border_index(y + j - half, src.height, border);
                acc += (sy < 0) ? 0.0 : tmp.at(sy, x);
            }
            out.at(y, x) = acc / k;
        }
    }
    return out;
}

Plane order_statistic(const Plane& src, int k, int rank, Border border) {
    if (k < 1) k = 1;
    const int total = k * k;
    rank = std::min(std::max(rank, 1), total);
    const int half = k / 2;
    std::vector<double> window(static_cast<std::size_t>(total));
    Plane out(src.width, src.height);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            std::size_t n = 0;
            for (int j = 0; j < k; ++j) {
                const int sy = border_index(y + j - half, src.height, border);
                for (int i = 0; i < k; ++i) {
                    const int sx = border_index(x + i - half, src.width, border);
                    window[n++] = (sy < 0 || sx < 0) ? 0.0 : src.at(sy, sx);
                }
            }
            std::sort(window.begin(), window.end());
            out.at(y, x) = window[static_cast<std::size_t>(rank - 1)];
        }
    }
    return out;
}

double percentile_of(std::vector<double> values, double pct) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double p = std::min(std::max(pct, 0.0), 100.0) / 100.0;
    const double pos = p * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

Plane positive_part(const Plane& p) {
    Plane out(p.width, p.height);
    double peak = 0.0;
    for (double v : p.values)
        if (std::isfinite(v) && v > peak) peak = v;
    if (peak <= 0.0) return out;
    for (std::size_t i = 0; i < p.values.size(); ++i) {
        const double v = std::isfinite(p.values[i]) ? p.values[i] : 0.0;
        out.values[i] = std::min(v, peak) / peak * 255.0;
    }
    return out;
}

}  // namespace diplab
