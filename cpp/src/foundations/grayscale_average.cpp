// I = (R + G + B) / 3
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params&) {
    Result r;
    r.image = plane_to_image(to_gray_plane(img, true));
    r.log =
        "s = (R + G + B) / 3\n"
        "Equal weight for every channel: simple, but it does not match\n"
        "human brightness perception.";
    return r;
}

}  // namespace

Operation op_grayscale_average() {
    Operation op;
    op.key = "gray_average";
    op.name = "Grayscale (average)";
    op.category = "Foundations";
    op.description = "RGB -> grayscale with the plain average (R + G + B) / 3.";
    op.run = run;
    return op;
}

}  // namespace diplab
