// Rank 1 order statistic.
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
    r.image = map_channels(img, [&](const Plane& p) { return order_statistic(p, k, 1, border); });
    r.log = "g(x,y) = min{ f(x+i, y+j) } over the k x k window\nk = " + std::to_string(k) +
            ", border = " + border_name +
            "\nThe minimum filter darkens: isolated bright points disappear while\n"
            "bright regions survive shrunken.";
    return r;
}

}  // namespace

Operation op_min_filter() {
    Operation op;
    op.key = "min";
    op.name = "Min filter";
    op.category = "Smoothing";
    op.description = "Smallest value in the neighbourhood (rank 1).";
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
