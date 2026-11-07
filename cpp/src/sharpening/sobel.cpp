// G = sqrt(Gx^2 + Gy^2) on the luminance plane.
#include <algorithm>

#include "diplab/operations.hpp"
#include "diplab/sharpening_common.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const std::string variant = params.text("variant", "Sobel");
    const double threshold = params.real("threshold", 25.0);

    const Gradient g = gradients(img, variant);
    const double peak = g.magnitude.maximum();

    Result r;
    if (peak <= 0.0) {
        r.image = Image(img.width, img.height, 1);
    } else {
        Plane scaled(img.width, img.height);
        for (std::size_t i = 0; i < scaled.values.size(); ++i) {
            double v = g.magnitude.values[i] / peak * 255.0;
            if (threshold > 0.0 && v < threshold) v = 0.0;
            scaled.values[i] = v;
        }
        r.image = plane_to_image(scaled);
    }

    const bool scharr = variant.rfind("Scharr", 0) == 0;
    r.aux.push_back({"Gradient direction", orientation_rgb(g.angle, g.magnitude)});
    r.log = variant +
            " first derivatives on the luminance plane:\n"
            "Gx = mask_x ⊛ f,   Gy = mask_y ⊛ f,   G = √(Gx² + Gy²)\n"
            "Flat regions give 0; edges give a large response.  The result is\n"
            "scaled to 0..255" +
            std::string(threshold > 0.0 ? " and thresholded." : ".") + "\n\nGx mask:\n" +
            describe_kernel(scharr ? kScharrX : kSobelX) + "\nGy mask:\n" +
            describe_kernel(scharr ? kScharrY : kSobelY);
    return r;
}

}  // namespace

Operation op_sobel() {
    Operation op;
    op.key = "sobel";
    op.name = "Sobel magnitude";
    op.category = "Sharpening";
    op.description = "First derivative: G = √(Gx² + Gy²). Bright edges on black.";
    op.run = run;
    Param variant;
    variant.key = "variant";
    variant.label = "Operator";
    variant.kind = "choice";
    variant.def = "Sobel";
    variant.choices = {"Sobel", "Scharr"};
    Param threshold;
    threshold.key = "threshold";
    threshold.label = "Threshold";
    threshold.kind = "float";
    threshold.def = "25";
    threshold.minimum = 0;
    threshold.maximum = 255;
    op.params = {variant, threshold};
    return op;
}

}  // namespace diplab
