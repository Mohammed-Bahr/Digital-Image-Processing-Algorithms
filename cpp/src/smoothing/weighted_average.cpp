// g = (1/16) * [[1,2,1],[2,4,2],[1,2,1]] correlated with f
#include "diplab/filters.hpp"
#include "diplab/operations.hpp"
#include "diplab/sharpening_common.hpp"

namespace diplab {
namespace {

const Kernel kWeighted3x3 = {{1.0 / 16, 2.0 / 16, 1.0 / 16},
                             {2.0 / 16, 4.0 / 16, 2.0 / 16},
                             {1.0 / 16, 2.0 / 16, 1.0 / 16}};

Result run(const Image& img, const Params& params) {
    const std::string border_name = params.text("border", "reflect");
    const Border border = parse_border(border_name);

    Result r;
    r.image = map_channels(img, [&](const Plane& p) { return correlate(p, kWeighted3x3, border); });
    r.log = "g = (1/16) · [[1,2,1],[2,4,2],[1,2,1]] ⊛ f\nborder = " + border_name +
            "\nThe centre counts four times an edge neighbour, so the weights fall\n"
            "off smoothly and this blurs less than a uniform 3x3 average.\n\n" +
            describe_kernel(kWeighted3x3);
    return r;
}

}  // namespace

Operation op_weighted_average() {
    Operation op;
    op.key = "weighted";
    op.name = "Weighted average 1/16";
    op.category = "Smoothing";
    op.description = "(1/16)·[[1,2,1],[2,4,2],[1,2,1]]: blurs less than a uniform average.";
    op.run = run;
    op.params = {choice_border()};
    return op;
}

}  // namespace diplab
