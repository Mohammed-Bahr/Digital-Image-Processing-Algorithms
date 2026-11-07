// Middle value of the sorted k x k neighbourhood.
#include "diplab/filters.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    int k = params.integer("k", 3);
    if (k < 1) k = 1;
    const std::string border_name = params.text("border", "reflect");
    const Border border = parse_border(border_name);
    const int rank = k * k / 2 + 1;

    Result r;
    r.image = map_channels(img, [&](const Plane& p) { return order_statistic(p, k, rank, border); });
    r.log = "Sort the k x k neighbourhood and keep the middle value.\nk = " +
            std::to_string(k) + "  (rank " + std::to_string(rank) + " of " +
            std::to_string(k * k) + "), border = " + border_name +
            "\nA single extreme pixel cannot move the median, so salt-and-pepper\n"
            "noise disappears while edges stay sharp.";
    return r;
}

}  // namespace

Operation op_median_filter() {
    Operation op;
    op.key = "median";
    op.name = "Median filter";
    op.category = "Smoothing";
    op.description = "Middle value of the sorted neighbourhood; removes salt-and-pepper noise.";
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
