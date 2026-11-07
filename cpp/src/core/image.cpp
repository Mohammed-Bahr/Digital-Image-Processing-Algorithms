#include "diplab/image.hpp"

#include <algorithm>
#include <cmath>

namespace diplab {

Plane::Plane(int w, int h, double fill)
    : width(w),
      height(h),
      values(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), fill) {}

double Plane::minimum() const {
    if (values.empty()) return 0.0;
    return *std::min_element(values.begin(), values.end());
}

double Plane::maximum() const {
    if (values.empty()) return 0.0;
    return *std::max_element(values.begin(), values.end());
}

Image::Image(int w, int h, int c)
    : width(w),
      height(h),
      channels(c),
      values(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * c, 0) {}

int Image::minimum() const {
    if (values.empty()) return 0;
    return *std::min_element(values.begin(), values.end());
}

int Image::maximum() const {
    if (values.empty()) return 0;
    return *std::max_element(values.begin(), values.end());
}

double Image::mean() const {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (std::uint8_t v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

std::uint8_t clamp_u8(double value) {
    if (!(value > 0.0)) return 0;  // also catches NaN
    if (value >= 255.0) return 255;
    return static_cast<std::uint8_t>(std::lround(value));
}

Plane channel_to_plane(const Image& img, int c) {
    Plane p(img.width, img.height);
    if (c < 0 || c >= img.channels) c = 0;
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
            p.at(y, x) = img.at(y, x, c);
    return p;
}

Plane to_gray_plane(const Image& img, bool average) {
    Plane p(img.width, img.height);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            if (img.channels == 1) {
                p.at(y, x) = img.at(y, x, 0);
            } else if (average) {
                p.at(y, x) = (img.at(y, x, 0) + img.at(y, x, 1) + img.at(y, x, 2)) / 3.0;
            } else {
                p.at(y, x) = 0.30 * img.at(y, x, 0) + 0.59 * img.at(y, x, 1) +
                             0.11 * img.at(y, x, 2);
            }
        }
    }
    return p;
}

Image plane_to_image(const Plane& p) {
    Image out(p.width, p.height, 1);
    for (int y = 0; y < p.height; ++y)
        for (int x = 0; x < p.width; ++x) out.at(y, x) = clamp_u8(p.at(y, x));
    return out;
}

Image planes_to_image(const std::vector<Plane>& planes) {
    if (planes.empty()) return Image();
    const int w = planes[0].width;
    const int h = planes[0].height;
    Image out(w, h, static_cast<int>(planes.size()));
    for (std::size_t c = 0; c < planes.size(); ++c) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                out.at(y, x, static_cast<int>(c)) = clamp_u8(planes[c].at(y, x));
    }
    return out;
}

Image to_gray_image(const Image& img) { return plane_to_image(to_gray_plane(img)); }

Image to_rgb_image(const Image& img) {
    if (img.channels == 3) return img;
    Image out(img.width, img.height, 3);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            const std::uint8_t v = img.at(y, x, 0);
            out.at(y, x, 0) = v;
            out.at(y, x, 1) = v;
            out.at(y, x, 2) = v;
        }
    return out;
}

Image apply_lut(const Image& img, const Lut& lut) {
    Image out(img.width, img.height, img.channels);
    for (std::size_t i = 0; i < img.values.size(); ++i)
        out.values[i] = lut[img.values[i]];
    return out;
}

Image map_channels(const Image& img, const std::function<Plane(const Plane&)>& fn) {
    Image out(img.width, img.height, img.channels);
    for (int c = 0; c < img.channels; ++c) {
        const Plane result = fn(channel_to_plane(img, c));
        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x)
                out.at(y, x, c) = clamp_u8(result.at(y, x));
    }
    return out;
}

Plane clamp_plane(const Plane& p, double lo, double hi) {
    Plane out(p.width, p.height);
    for (std::size_t i = 0; i < p.values.size(); ++i)
        out.values[i] = std::min(std::max(p.values[i], lo), hi);
    return out;
}

void normalize_minmax(Plane& p, double lo, double hi) {
    if (p.values.empty()) return;
    const double mn = p.minimum();
    const double mx = p.maximum();
    if (mx - mn < 1e-12) {
        const double mid = std::lround((lo + hi) / 2.0);
        for (double& v : p.values) v = mid;
        return;
    }
    const double scale = (hi - lo) / (mx - mn);
    for (double& v : p.values) v = (v - mn) * scale + lo;
}

Plane resize_bilinear(const Plane& src, int out_w, int out_h) {
    Plane out(out_w, out_h);
    if (src.empty() || out_w <= 0 || out_h <= 0) return out;
    for (int y = 0; y < out_h; ++y) {
        double sy = (y + 0.5) * src.height / out_h - 0.5;
        if (sy < 0.0) sy = 0.0;
        int y0 = static_cast<int>(std::floor(sy));
        if (y0 > src.height - 1) y0 = src.height - 1;
        int y1 = std::min(y0 + 1, src.height - 1);
        const double fy = sy - y0;
        for (int x = 0; x < out_w; ++x) {
            double sx = (x + 0.5) * src.width / out_w - 0.5;
            if (sx < 0.0) sx = 0.0;
            int x0 = static_cast<int>(std::floor(sx));
            if (x0 > src.width - 1) x0 = src.width - 1;
            int x1 = std::min(x0 + 1, src.width - 1);
            const double fx = sx - x0;
            const double top = src.at(y0, x0) * (1.0 - fx) + src.at(y0, x1) * fx;
            const double bot = src.at(y1, x0) * (1.0 - fx) + src.at(y1, x1) * fx;
            out.at(y, x) = top * (1.0 - fy) + bot * fy;
        }
    }
    return out;
}

Lut identity_lut() {
    Lut lut{};
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<std::uint8_t>(i);
    return lut;
}

}  // namespace diplab
