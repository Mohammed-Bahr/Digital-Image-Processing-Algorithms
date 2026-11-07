// Shared kernels and helpers for the derivative-based algorithms.
#pragma once

#include <string>

#include "diplab/filters.hpp"
#include "diplab/image.hpp"

namespace diplab {

inline const Kernel kSobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
inline const Kernel kSobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
inline const Kernel kScharrX = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
inline const Kernel kScharrY = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};

bool is_laplacian_kind(const std::string& kind);
Kernel laplacian_kernel(const std::string& kind);  // throws on unknown kind

struct Gradient {
    Plane gx, gy, magnitude, angle;
};

Gradient gradients(const Image& img, const std::string& variant,
                   Border border = Border::Reflect);

// Hue = angle, brightness = |G| / max|G|.
Image orientation_rgb(const Plane& angle, const Plane& magnitude);

// |Laplacian| of the luminance plane, scaled to 0..255.
Plane laplacian_abs(const Image& img, const std::string& kind,
                    Border border = Border::Reflect);

std::string describe_kernel(const Kernel& kernel);

}  // namespace diplab
