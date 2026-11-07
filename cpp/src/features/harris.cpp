// R = det(M) - k * trace(M)^2, then NMS and minimum-distance thinning.
#include <algorithm>
#include <utility>
#include <vector>

#include "diplab/filters.hpp"
#include "diplab/harris.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Image draw_markers(const Image& img, const std::vector<std::pair<int, int>>& points,
                   int radius) {
    Image out = to_rgb_image(img);
    const int r = std::max(1, radius);
    for (const auto& [py, px] : points) {
        for (int dy = -r - 1; dy <= r + 1; ++dy) {
            for (int dx = -r - 1; dx <= r + 1; ++dx) {
                const double d2 = static_cast<double>(dy * dy + dx * dx);
                const bool inside = d2 <= r * r;
                const bool ring = d2 <= (r + 1.0) * (r + 1.0) && d2 > r * r;
                if (!inside && !ring) continue;
                const int y = py + dy;
                const int x = px + dx;
                if (y < 0 || y >= out.height || x < 0 || x >= out.width) continue;
                if (inside) {
                    out.at(y, x, 0) = 255;
                    out.at(y, x, 1) = 60;
                    out.at(y, x, 2) = 60;
                } else {
                    out.at(y, x, 0) = 255;
                    out.at(y, x, 1) = 255;
                    out.at(y, x, 2) = 255;
                }
            }
        }
    }
    return out;
}

Result run(const Image& img, const Params& params) {
    const double k = params.real("k", 0.04);
    const int window = params.integer("window", 3);
    const double pct = params.real("threshold", 99.8);
    const int min_distance = params.integer("min_distance", 8);
    const int radius = params.integer("radius", 5);

    Plane response;
    const auto points = harris_corners(img, k, window, pct, min_distance, 400, &response);

    Result r;
    r.image = draw_markers(img, points, radius);
    r.aux.push_back({"Harris response R", plane_to_image(positive_part(response))});
    r.meta["corners"] = static_cast<double>(points.size());
    r.log = "R = det(M) - k · trace(M)² = λ1λ2 - k(λ1 + λ2)²\nk = " +
            std::to_string(k) + ", window = " + std::to_string(window) + "x" +
            std::to_string(window) + ", threshold = " + std::to_string(pct) +
            "th percentile, min distance = " + std::to_string(min_distance) +
            " px.\nCorners found: " + std::to_string(points.size()) +
            "\nCorner: both eigenvalues large -> R positive.\n"
            "Edge: one large, one small -> R negative.\nFlat: both small -> R ≈ 0.";
    return r;
}

}  // namespace

Operation op_harris() {
    Operation op;
    op.key = "harris";
    op.name = "Harris corners";
    op.category = "Features";
    op.description = "Detect corners: R = det(M) - k·trace(M)².";
    op.run = run;
    Param k;
    k.key = "k";
    k.label = "k";
    k.kind = "float";
    k.def = "0.04";
    k.minimum = 0.01;
    k.maximum = 0.10;
    k.step = 0.005;
    Param window;
    window.key = "window";
    window.label = "Window";
    window.kind = "int";
    window.def = "3";
    window.minimum = 3;
    window.maximum = 11;
    window.step = 2;
    Param threshold;
    threshold.key = "threshold";
    threshold.label = "Threshold percentile";
    threshold.kind = "float";
    threshold.def = "99.8";
    threshold.minimum = 80;
    threshold.maximum = 99.9;
    threshold.step = 0.1;
    Param min_distance;
    min_distance.key = "min_distance";
    min_distance.label = "Min distance (px)";
    min_distance.kind = "int";
    min_distance.def = "8";
    min_distance.minimum = 1;
    min_distance.maximum = 40;
    Param radius;
    radius.key = "radius";
    radius.label = "Marker radius";
    radius.kind = "int";
    radius.def = "5";
    radius.minimum = 2;
    radius.maximum = 20;
    op.params = {k, window, threshold, min_distance, radius};
    return op;
}

}  // namespace diplab
