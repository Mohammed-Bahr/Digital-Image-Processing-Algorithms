// Spatial primitives: border handling, 2-D correlation, order statistics.
#pragma once

#include <string>
#include <vector>

#include "diplab/image.hpp"

namespace diplab {

enum class Border { Reflect, Replicate, Zeros };

Border parse_border(const std::string& name);         // "reflect" is the default
int border_index(int i, int n, Border mode);          // maps i into [0, n)

using Kernel = std::vector<std::vector<double>>;       // odd x odd

// g(x,y) = sum_{s,t} w(s,t) * f(x+s, y+t)   (correlation, not convolution)
Plane correlate(const Plane& src, const Kernel& kernel, Border border);

// Separable k x k mean (coefficients 1/k^2).
Plane box_mean(const Plane& src, int k, Border border);

// rank is 1-based: 1 = min, k^2 = max.
Plane order_statistic(const Plane& src, int k, int rank, Border border);

// Linear-interpolated percentile, matching numpy's default.
double percentile_of(std::vector<double> values, double pct);

// Non-negative part of a map, stretched to 0..255 for display.
Plane positive_part(const Plane& p);

}  // namespace diplab
