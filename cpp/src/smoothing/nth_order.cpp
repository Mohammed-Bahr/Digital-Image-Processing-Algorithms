// n-th value of the sorted k x k window (1-based).
#include <algorithm>

#include "diplab/filters.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    int k = params.integer("k", 3);
    if (k < 1) k = 1;
    const int max_rank = k * k;
    const int rank = std::min(std::max(params.integer("rank", 5), 1), max_rank);
    const std::string border_name = params.text("border", "reflect");
    const Border border = parse_border(border_name);

    std::string name = "order statistic";
    if (rank == 1)
        name = "minimum";
    else if (rank == max_rank)
        name = "maximum";
    else if (rank == max_rank / 2 + 1)
        name = "median";

    Result r;
    r.image = map_channels(img, [&](const Plane& p) { return order_statistic(p, k, rank, border); });
    r.log = "Sort the k x k window and take the n-th value.\nk = " + std::to_string(k) +
            ", n = " + std::to_string(rank) + " of " + std::to_string(max_rank) +
            ", border = " + border_name + "\nThis is the " + name +
            " filter: small ranks suppress bright spots, large ranks\n"
            "suppress dark spots, the median removes impulses without moving edges.";
    return r;
}

}  // namespace

Operation op_nth_order() {
    Operation op;
    op.key = "nth";
    op.name = "Nth-order statistic";
    op.category = "Smoothing";
    op.description = "Pick the n-th value after sorting (1 = min, middle = median, k² = max).";
    op.run = run;
    Param k;
    k.key = "k";
    k.label = "Kernel size";
    k.kind = "int";
    k.def = "3";
    k.minimum = 3;
    k.maximum = 15;
    k.step = 2;
    Param rank;
    rank.key = "rank";
    rank.label = "Rank (1..k²)";
    rank.kind = "int";
    rank.def = "5";
    rank.minimum = 1;
    rank.maximum = 225;
    op.params = {k, rank, choice_border()};
    return op;
}

}  // namespace diplab
