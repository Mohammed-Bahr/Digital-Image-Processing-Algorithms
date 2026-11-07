// s = c * r^gamma, with r normalised to 0..1
#include <cmath>

#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double gamma = std::max(params.real("gamma", 0.6), 1e-6);
    const double c = params.real("c", 1.0);

    Lut lut{};
    for (int i = 0; i < 256; ++i) {
        const double r = i / 255.0;
        lut[i] = clamp_u8(255.0 * c * std::pow(r, gamma));
    }

    std::string effect;
    if (gamma < 1.0)
        effect = "γ < 1 brightens the image (expands dark values).";
    else if (gamma > 1.0)
        effect = "γ > 1 darkens the image (compresses dark values).";
    else
        effect = "γ = 1 leaves the image unchanged.";

    Result res;
    res.image = apply_lut(img, lut);
    res.log = "s = c · r^γ,   γ = " + std::to_string(gamma) + ", c = " + std::to_string(c) +
              "\n" + effect;
    return res;
}

}  // namespace

Operation op_gamma() {
    Operation op;
    op.key = "gamma";
    op.name = "Power-law (gamma)";
    op.category = "Point Processing";
    op.description = "s = c · r^γ after normalising r to 0..1.";
    op.run = run;
    Param g;
    g.key = "gamma";
    g.label = "γ";
    g.kind = "float";
    g.def = "0.6";
    g.minimum = 0.05;
    g.maximum = 5.0;
    g.step = 0.05;
    Param c;
    c.key = "c";
    c.label = "c";
    c.kind = "float";
    c.def = "1.0";
    c.minimum = 0.0;
    c.maximum = 3.0;
    c.step = 0.05;
    op.params = {g, c};
    return op;
}

}  // namespace diplab
