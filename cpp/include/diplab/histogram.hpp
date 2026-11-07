// Histogram equalization LUT: s_k = round((L - 1) * CDF(k)).
#pragma once

#include "diplab/image.hpp"

namespace diplab {

// Deliberately does not subtract cdf_min, matching the notes' worked example.
Lut equalization_lut(const Plane& channel, int levels = 256);

}  // namespace diplab
