// s = (L - 1) - r
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params&) {
    Lut lut{};
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<std::uint8_t>(255 - i);
    Result r;
    r.image = apply_lut(img, lut);
    r.log =
        "s = (L - 1) - r = 255 - r\n"
        "Dark becomes light and light becomes dark.  The transfer curve is a\n"
        "straight line with slope -1.";
    return r;
}

}  // namespace

Operation op_negative() {
    Operation op;
    op.key = "negative";
    op.name = "Negative";
    op.category = "Point Processing";
    op.description = "s = (L - 1) - r. Reverses brightness.";
    op.run = run;
    return op;
}

}  // namespace diplab
