// Rank k^2 order statistic.
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
    r.image = map_channels(img, [&](const Plane& p) { return order_statistic(p, k, k * k, border); });
    r.log = "g(x,y) = max{ f(x+i, y+j) } over the k x k window\nk = " + std::to_string(k) +
            ", border = " + border_name +
            "\nThe maximum filter brightens: dark isolated points disappear and\n"
            "bright regions grow.";
    return r;
}

}  // namespace

Operation op_max_filter() {
    Operation op;
    op.key = "max";
    op.name = "Max filter";
    op.category = "Smoothing";
    op.description = "Largest value in the neighbourhood (rank k²).";
    op.run = run;
    Param k;
    k.key = "k";
    k.label = "Kernel size";
    k.kind = "int";
    k.def = "3";
    k.minimum = 3;
    k.maximum = 15;
    k.step = 2;
    op.params = {k, choice_border()};
    return op;
}

}  // namespace diplab
