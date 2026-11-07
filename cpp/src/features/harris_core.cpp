#include "diplab/harris.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "diplab/filters.hpp"
#include "diplab/sharpening_common.hpp"

namespace diplab {

Plane harris_response_map(const Image& img, double k, int window) {
    window = std::max(window, 3);
    if (window % 2 == 0) ++window;

    const Plane gray = to_gray_plane(img);
    const Plane gx = correlate(gray, kSobelX, Border::Reflect);
    const Plane gy = correlate(gray, kSobelY, Border::Reflect);

    Plane ixx(gray.width, gray.height);
    Plane iyy(gray.width, gray.height);
    Plane ixy(gray.width, gray.height);
    for (std::size_t i = 0; i < gray.values.size(); ++i) {
        ixx.values[i] = gx.values[i] * gx.values[i];
        iyy.values[i] = gy.values[i] * gy.values[i];
        ixy.values[i] = gx.values[i] * gy.values[i];
    }

    const Plane sxx = box_mean(ixx, window, Border::Reflect);
    const Plane syy = box_mean(iyy, window, Border::Reflect);
    const Plane sxy = box_mean(ixy, window, Border::Reflect);

    Plane response(gray.width, gray.height);
    for (std::size_t i = 0; i < response.values.size(); ++i) {
        const double det = sxx.values[i] * syy.values[i] - sxy.values[i] * sxy.values[i];
        const double trace = sxx.values[i] + syy.values[i];
        response.values[i] = det - k * trace * trace;
    }
    return response;
}

std::vector<std::pair<int, int>> harris_corners(const Image& img, double k, int window,
                                                double threshold_pct, int min_distance,
                                                int max_points, Plane* response_out) {
    const Plane r = harris_response_map(img, k, window);
    if (response_out != nullptr) *response_out = r;
    if (r.values.empty()) return {};

    const double threshold = percentile_of(r.values, threshold_pct);
    std::vector<std::pair<double, std::size_t>> candidates;

    for (int y = 0; y < r.height; ++y) {
        for (int x = 0; x < r.width; ++x) {
            const double v = r.at(y, x);
            if (!(v > threshold) || v <= 0.0) continue;
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int sy = y + dy;
                    const int sx = x + dx;
                    if (sy < 0 || sy >= r.height || sx < 0 || sx >= r.width) continue;
                    if (r.at(sy, sx) > v + 1e-9) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (is_max) candidates.emplace_back(v, static_cast<std::size_t>(y) * r.width + x);
        }
    }
    if (candidates.empty()) return {};

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    const int min_dist = std::max(min_distance, 1);
    const double min_dist2 = static_cast<double>(min_dist) * min_dist;
    std::vector<std::pair<int, int>> points;
    for (const auto& [value, index] : candidates) {
        (void)value;
        const int y = static_cast<int>(index / r.width);
        const int x = static_cast<int>(index % r.width);
        bool far_enough = true;
        for (const auto& [py, px] : points) {
            const double dy = y - py;
            const double dx = x - px;
            if (dy * dy + dx * dx < min_dist2) {
                far_enough = false;
                break;
            }
        }
        if (!far_enough) continue;
        points.emplace_back(y, x);
        if (static_cast<int>(points.size()) >= max_points) break;
    }
    return points;
}

}  // namespace diplab
