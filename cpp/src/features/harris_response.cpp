// Positive part of R, stretched to 0..255.
#include "diplab/filters.hpp"
#include "diplab/harris.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double k = params.real("k", 0.04);
    const int window = params.integer("window", 3);
    const Plane response = harris_response_map(img, k, window);

    Result r;
    r.image = plane_to_image(positive_part(response));
    r.log = "Harris response R = det(M) - k·trace(M)², positive part rescaled to 0..255.\n"
            "k = " +
            std::to_string(k) + ", window = " + std::to_string(window) +
            "x" + std::to_string(window) +
            ".\nBright pixels are corner candidates; dark pixels are flat or edge-like.";
    return r;
}

}  // namespace

Operation op_harris_response() {
    Operation op;
    op.key = "harris_map";
    op.name = "Harris response map";
    op.category = "Features";
    op.description = "The raw R map: bright pixels are corner candidates.";
    op.run = run;
    Param k;
    k.key = "k";
    k.label = "k";
    k.kind = "float";
    k.def = "0.04";
    k.minimum = 0.01;
    k.maximum = 0.10;
    k.step = 0.005;
    Param window;
    window.key = "window";
    window.label = "Window";
    window.kind = "int";
    window.def = "3";
    window.minimum = 3;
    window.maximum = 11;
    window.step = 2;
    op.params = {k, window};
    return op;
}

}  // namespace diplab
