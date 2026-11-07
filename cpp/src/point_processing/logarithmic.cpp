// s = c * log(1 + r)
#include <cmath>

#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    double c = params.real("c", 0.0);
    std::string note;
    if (c <= 0.0) {
        double peak = 0.0;
        for (std::uint8_t v : img.values) peak = std::max(peak, static_cast<double>(v));
        c = 255.0 / std::log1p(std::max(peak, 1.0));
        note = " (auto: c = 255 / log(1 + r_max))";
    }

    Lut lut{};
    for (int i = 0; i < 256; ++i)
        lut[i] = clamp_u8(c * std::log1p(static_cast<double>(i)));

    Result r;
    r.image = apply_lut(img, lut);
    r.log = "s = c · log(1 + r),   c = " + std::to_string(c) + note +
            "\nThe curve is steep at low r and flat at high r: dark detail is\n"
            "expanded, bright detail is compressed.";
    return r;
}

}  // namespace

Operation op_logarithmic() {
    Operation op;
    op.key = "log";
    op.name = "Logarithmic";
    op.category = "Point Processing";
    op.description = "s = c · log(1 + r). Expands dark values, compresses bright ones.";
    op.run = run;
    Param p;
    p.key = "c";
    p.label = "c (0 = auto)";
    p.kind = "float";
    p.def = "0";
    p.minimum = 0;
    p.maximum = 10;
    p.step = 0.05;
    op.params = {p};
    return op;
}

}  // namespace diplab
