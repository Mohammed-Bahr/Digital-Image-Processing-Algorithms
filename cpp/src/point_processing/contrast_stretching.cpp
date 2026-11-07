// X_new = (X - Xmin) / (Xmax - Xmin) * 255, with optional percentiles.
#include <string>
#include <vector>

#include "diplab/filters.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double low = params.real("low", 0.0);
    const double high = params.real("high", 100.0);

    std::vector<double> values;
    values.reserve(img.values.size());
    for (std::uint8_t v : img.values) values.push_back(v);

    const double lo = percentile_of(values, low);
    const double hi = percentile_of(values, high);

    Result r;
    if (hi <= lo) {
        r.image = img;
        r.log = "Percentiles " + std::to_string(low) + "%.." + std::to_string(high) +
                "% give an empty range, so the image was left unchanged.";
        return r;
    }
    Image out(img.width, img.height, img.channels);
    for (std::size_t i = 0; i < img.values.size(); ++i)
        out.values[i] = clamp_u8((img.values[i] - lo) / (hi - lo) * 255.0);

    r.image = out;
    r.log = "X_new = (X - X_min) / (X_max - X_min) · 255\n"
            "X_min = " + std::to_string(lo) + ",  X_max = " + std::to_string(hi) +
            "\nA pure min-max stretch uses 0% and 100%; moving the percentiles\n"
            "inward clips outliers so they cannot ruin the range.";
    return r;
}

}  // namespace

Operation op_contrast_stretching() {
    Operation op;
    op.key = "contrast";
    op.name = "Contrast stretching";
    op.category = "Point Processing";
    op.description = "X_new = (X - Xmin)/(Xmax - Xmin) · 255, optional percentile clipping.";
    op.run = run;
    Param low;
    low.key = "low";
    low.label = "Low percentile";
    low.kind = "float";
    low.def = "0";
    low.minimum = 0;
    low.maximum = 49;
    low.step = 0.5;
    Param high;
    high.key = "high";
    high.label = "High percentile";
    high.kind = "float";
    high.def = "100";
    high.minimum = 51;
    high.maximum = 100;
    high.step = 0.5;
    op.params = {low, high};
    return op;
}

}  // namespace diplab
