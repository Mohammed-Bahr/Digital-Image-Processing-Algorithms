// g(x,y) = (1/k^2) * sum of the k x k neighbourhood
#include "diplab/filters.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    int k = params.integer("k", 3);
    if (k < 1) k = 1;
    const std::string border_name = params.text("border", "reflect");
    const Border border = parse_border(border_name);

    Result r;
    r.image = map_channels(img, [&](const Plane& p) { return box_mean(p, k, border); });
    r.log = "g(x,y) = (1/k²) · Σ f(x+i, y+j)   over the k x k window\nk = " +
            std::to_string(k) + ", border = " + border_name +
            "\nEvery neighbour has the same weight and the coefficients sum to 1,\n"
            "so the average brightness is preserved.  Bigger k removes more noise\n"
            "but blurs edges more.";
    return r;
}

}  // namespace

Operation op_box_filter() {
    Operation op;
    op.key = "box";
    op.name = "Average (box) filter";
    op.category = "Smoothing";
    op.description = "Uniform k x k average. Coefficients sum to 1, so brightness is preserved.";
    op.run = run;
    Param k;
    k.key = "k";
    k.label = "Kernel size";
    k.kind = "int";
    k.def = "3";
    k.minimum = 3;
    k.maximum = 31;
    k.step = 2;
    Param border = choice_border();
    op.params = {k, border};
    return op;
}

}  // namespace diplab
