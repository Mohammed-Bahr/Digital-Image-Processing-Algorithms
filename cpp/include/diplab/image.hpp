// Minimal image types: an 8-bit Image and a double Plane (single channel).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

namespace diplab {

using Lut = std::array<std::uint8_t, 256>;

// One channel of floating point samples, row-major.
struct Plane {
    int width = 0;
    int height = 0;
    std::vector<double> values;

    Plane() = default;
    Plane(int w, int h, double fill = 0.0);

    double& at(int y, int x) {
        return values[static_cast<std::size_t>(y) * width + x];
    }
    double at(int y, int x) const {
        return values[static_cast<std::size_t>(y) * width + x];
    }

    bool empty() const { return width <= 0 || height <= 0; }
    std::size_t size() const { return values.size(); }
    double minimum() const;
    double maximum() const;
};

// 1 (gray) or 3 (RGB) channels, interleaved, row-major.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<std::uint8_t> values;

    Image() = default;
    Image(int w, int h, int c = 1);

    std::uint8_t& at(int y, int x, int c = 0) {
        return values[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
    std::uint8_t at(int y, int x, int c = 0) const {
        return values[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }

    bool is_gray() const { return channels == 1; }
    bool empty() const { return width <= 0 || height <= 0; }
    int minimum() const;
    int maximum() const;
    double mean() const;
};

std::uint8_t clamp_u8(double value);

Plane channel_to_plane(const Image& img, int c);
Plane to_gray_plane(const Image& img, bool average = false);  // 0.30/0.59/0.11 or mean
Image plane_to_image(const Plane& p);                         // clamp + round
Image planes_to_image(const std::vector<Plane>& planes);
Image to_gray_image(const Image& img);
Image to_rgb_image(const Image& img);
Image apply_lut(const Image& img, const Lut& lut);

// Apply fn to each channel independently (or to the single gray plane).
Image map_channels(const Image& img, const std::function<Plane(const Plane&)>& fn);

Plane clamp_plane(const Plane& p, double lo, double hi);
void normalize_minmax(Plane& p, double lo = 0.0, double hi = 255.0);
Plane resize_bilinear(const Plane& src, int out_w, int out_h);
Lut identity_lut();

}  // namespace diplab
