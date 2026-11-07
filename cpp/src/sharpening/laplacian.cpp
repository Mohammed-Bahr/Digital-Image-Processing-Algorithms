// |Laplacian f| scaled to 0..255.
#include <stdexcept>

#include "diplab/operations.hpp"
#include "diplab/sharpening_common.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const std::string kind = params.text("kind", "4-neighbor");
    if (!is_laplacian_kind(kind))
        throw std::invalid_argument("unknown Laplacian mask: " + kind);

    Result r;
    r.image = plane_to_image(laplacian_abs(img, kind));
    r.log = "Second derivative ∇²f, shown as |∇²f| scaled to 0..255.\nMask: " + kind +
            "\nThe coefficients sum to zero, so a constant region gives zero.\n"
            "Edges and isolated points give a strong response.\n\n" +
            describe_kernel(laplacian_kernel(kind));
    return r;
}

}  // namespace

Operation op_laplacian() {
    Operation op;
    op.key = "laplacian";
    op.name = "Laplacian response";
    op.category = "Sharpening";
    op.description = "Second derivative |∇²f|; coefficients sum to zero.";
    op.run = run;
    Param kind;
    kind.key = "kind";
    kind.label = "Mask";
    kind.kind = "choice";
    kind.def = "4-neighbor";
    kind.choices = {"4-neighbor", "8-neighbor", "diagonal"};
    op.params = {kind};
    return op;
}

}  // namespace diplab
