// Three linear segments through (r1,s1) and (r2,s2).
#include <stdexcept>

#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double r1 = params.real("r1", 70);
    const double s1 = params.real("s1", 30);
    const double r2 = params.real("r2", 180);
    const double s2 = params.real("s2", 225);
    if (!(1.0 <= r1 && r1 < r2 && r2 <= 254.0))
        throw std::invalid_argument("control points need 1 <= r1 < r2 <= 254");

    Lut lut{};
    for (int i = 0; i < 256; ++i) {
        const double x = i;
        double y = 0.0;
        if (x <= r1)
            y = (s1 / r1) * x;
        else if (x <= r2)
            y = ((s2 - s1) / (r2 - r1)) * (x - r1) + s1;
        else
            y = ((255.0 - s2) / (255.0 - r2)) * (x - r2) + s2;
        lut[i] = clamp_u8(y);
    }

    Result r;
    r.image = apply_lut(img, lut);
    r.log = "Three linear segments through (0,0) -> (" + std::to_string(r1) + "," +
            std::to_string(s1) + ") -> (" + std::to_string(r2) + "," + std::to_string(s2) +
            ") -> (255,255)\n"
            "A steep middle segment stretches the mid-range; shallow end\n"
            "segments compress shadows and highlights.";
    return r;
}

}  // namespace

Operation op_piecewise_linear() {
    Operation op;
    op.key = "piecewise";
    op.name = "Piecewise linear";
    op.category = "Point Processing";
    op.description = "Three linear segments through two control points.";
    op.run = run;
    Param r1;
    r1.key = "r1";
    r1.label = "r1";
    r1.kind = "int";
    r1.def = "70";
    r1.minimum = 1;
    r1.maximum = 250;
    Param s1;
    s1.key = "s1";
    s1.label = "s1";
    s1.kind = "int";
    s1.def = "30";
    s1.minimum = 0;
    s1.maximum = 255;
    Param r2;
    r2.key = "r2";
    r2.label = "r2";
    r2.kind = "int";
    r2.def = "180";
    r2.minimum = 2;
    r2.maximum = 254;
    Param s2;
    s2.key = "s2";
    s2.label = "s2";
    s2.kind = "int";
    s2.def = "225";
    s2.minimum = 0;
    s2.maximum = 255;
    op.params = {r1, s1, r2, s2};
    return op;
}

}  // namespace diplab
