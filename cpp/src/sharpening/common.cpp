#include "diplab/sharpening_common.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "diplab/fft.hpp"

namespace diplab {

bool is_laplacian_kind(const std::string& kind) {
    return kind == "4-neighbor" || kind == "8-neighbor" || kind == "diagonal";
}

Kernel laplacian_kernel(const std::string& kind) {
    if (kind == "4-neighbor") return {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    if (kind == "8-neighbor") return {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
    if (kind == "diagonal") return {{-1, 2, -1}, {2, -4, 2}, {-1, 2, -1}};
    throw std::invalid_argument("unknown Laplacian mask: " + kind);
}

Gradient gradients(const Image& img, const std::string& variant, Border border) {
    const Plane gray = to_gray_plane(img);
    const bool scharr = variant.rfind("Scharr", 0) == 0;
    Gradient g;
    g.gx = correlate(gray, scharr ? kScharrX : kSobelX, border);
    g.gy = correlate(gray, scharr ? kScharrY : kSobelY, border);
    g.magnitude = Plane(gray.width, gray.height);
    g.angle = Plane(gray.width, gray.height);
    for (std::size_t i = 0; i < gray.values.size(); ++i) {
        const double gx = g.gx.values[i];
        const double gy = g.gy.values[i];
        g.magnitude.values[i] = std::hypot(gx, gy);
        g.angle.values[i] = std::atan2(gy, gx);
    }
    return g;
}

Image orientation_rgb(const Plane& angle, const Plane& magnitude) {
    const double peak = magnitude.maximum();
    Image out(angle.width, angle.height, 3);
    for (int y = 0; y < angle.height; ++y) {
        for (int x = 0; x < angle.width; ++x) {
            const double strength = peak > 0.0 ? magnitude.at(y, x) / peak : 0.0;
            const double a = angle.at(y, x);
            const double r = 0.5 + 0.5 * std::cos(a);
            const double g = 0.5 + 0.5 * std::cos(a - 2.0 * kPi / 3.0);
            const double b = 0.5 + 0.5 * std::cos(a + 2.0 * kPi / 3.0);
            out.at(y, x, 0) = clamp_u8(r * strength * 255.0);
            out.at(y, x, 1) = clamp_u8(g * strength * 255.0);
            out.at(y, x, 2) = clamp_u8(b * strength * 255.0);
        }
    }
    return out;
}

Plane laplacian_abs(const Image& img, const std::string& kind, Border border) {
    const Plane lap = correlate(to_gray_plane(img), laplacian_kernel(kind), border);
    Plane out(lap.width, lap.height);
    const double peak = std::max(std::fabs(lap.minimum()), std::fabs(lap.maximum()));
    if (peak <= 0.0) return out;
    for (std::size_t i = 0; i < lap.values.size(); ++i)
        out.values[i] = std::fabs(lap.values[i]) / peak * 255.0;
    return out;
}

std::string describe_kernel(const Kernel& kernel) {
    if (kernel.empty()) return "";
    std::ostringstream out;
    out << std::setprecision(4);
    for (const auto& row : kernel) {
        out << "[ ";
        for (double v : row) out << std::setw(8) << v << " ";
        out << "]\n";
    }
    return out.str();
}

}  // namespace diplab
