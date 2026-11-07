// Harris corner detection, written from the formulas in the notes.
#pragma once

#include <utility>
#include <vector>

#include "diplab/image.hpp"

namespace diplab {

// R = det(M) - k * trace(M)^2 on the luminance plane.
Plane harris_response_map(const Image& img, double k = 0.04, int window = 3);

// Corners as (row, col) after 3x3 non-maximum suppression, a percentile
// threshold and minimum-distance thinning.
std::vector<std::pair<int, int>> harris_corners(const Image& img, double k,
                                                int window,
                                                double threshold_pct,
                                                int min_distance,
                                                int max_points = 400,
                                                Plane* response_out = nullptr);

}  // namespace diplab
