// theta = atan2(Gy, Gx) encoded as hue.
#include "diplab/operations.hpp"
#include "diplab/sharpening_common.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const std::string variant = params.text("variant", "Sobel");
    const Gradient g = gradients(img, variant);

    Plane mag = g.magnitude;
    normalize_minmax(mag);

    Result r;
    r.image = orientation_rgb(g.angle, g.magnitude);
    r.aux.push_back({"Gradient magnitude", plane_to_image(mag)});
    r.log = variant +
            " gradient direction: θ = atan2(Gy, Gx)\n"
            "Hue encodes θ over -π..π, brightness encodes |G|.  The gradient\n"
            "points across an edge, so the colour gives the edge orientation.";
    return r;
}

}  // namespace

Operation op_sobel_orientation() {
    Operation op;
    op.key = "sobel_dir";
    op.name = "Sobel orientation";
    op.category = "Sharpening";
    op.description = "θ = atan2(Gy, Gx). Hue shows direction, brightness shows magnitude.";
    op.run = run;
    Param variant;
    variant.key = "variant";
    variant.label = "Operator";
    variant.kind = "choice";
    variant.def = "Sobel";
    variant.choices = {"Sobel", "Scharr"};
    op.params = {variant};
    return op;
}

}  // namespace diplab
