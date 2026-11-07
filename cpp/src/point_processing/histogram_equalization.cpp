// s_k = round((L - 1) * CDF(k)), without subtracting cdf_min (as in the notes).
#include <algorithm>
#include <array>
#include <cmath>

#include "diplab/histogram.hpp"
#include "diplab/operations.hpp"

namespace diplab {

Lut equalization_lut(const Plane& channel, int levels) {
    levels = std::min(std::max(levels, 2), 256);
    std::array<double, 256> counts{};
    for (double v : channel.values) {
        int k = static_cast<int>(std::lround(v));
        k = std::min(std::max(k, 0), levels - 1);
        counts[static_cast<std::size_t>(k)] += 1.0;
    }
    double total = 0.0;
    for (int i = 0; i < levels; ++i) total += counts[static_cast<std::size_t>(i)];

    Lut lut = identity_lut();
    if (total <= 0.0) return lut;

    double cdf = 0.0;
    for (int i = 0; i < levels; ++i) {
        cdf += counts[static_cast<std::size_t>(i)] / total;
        lut[static_cast<std::size_t>(i)] = clamp_u8(std::round((levels - 1) * cdf));
    }
    return lut;
}

namespace {

Result run(const Image& img, const Params&) {
    Image out(img.width, img.height, img.channels);
    for (int c = 0; c < img.channels; ++c) {
        const Lut lut = equalization_lut(channel_to_plane(img, c));
        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x)
                out.at(y, x, c) = lut[img.at(y, x, c)];
    }

    Result r;
    r.image = out;
    r.log =
        "s_k = round((L - 1) · CDF(k)),   L = 256\n"
        "The CDF is normalised to 0..1 and stretched across the full range, so\n"
        "the output histogram is approximately uniform and the contrast rises.";
    return r;
}

}  // namespace

Operation op_histogram_equalization() {
    Operation op;
    op.key = "equalize";
    op.name = "Histogram equalization";
    op.category = "Point Processing";
    op.description = "Remap intensities through the CDF: s_k = round((L-1)·CDF(k)).";
    op.run = run;
    return op;
}

}  // namespace diplab
