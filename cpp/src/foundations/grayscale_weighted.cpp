// I = 0.30 R + 0.59 G + 0.11 B
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params&) {
    Result r;
    r.image = plane_to_image(to_gray_plane(img, false));
    r.log =
        "s = 0.30 R + 0.59 G + 0.11 B\n"
        "The eye is most sensitive to green and least to blue, so green gets\n"
        "the largest coefficient.";
    return r;
}

}  // namespace

Operation op_grayscale_weighted() {
    Operation op;
    op.key = "gray_weighted";
    op.name = "Grayscale (weighted)";
    op.category = "Foundations";
    op.description = "RGB -> grayscale with perceptual weights 0.30 R + 0.59 G + 0.11 B.";
    op.run = run;
    return op;
}

}  // namespace diplab
