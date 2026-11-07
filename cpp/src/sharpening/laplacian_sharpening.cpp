// g = f - c * Laplacian(f)
#include <stdexcept>
#include <vector>

#include "diplab/operations.hpp"
#include "diplab/sharpening_common.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const std::string kind = params.text("kind", "4-neighbor");
    const double strength = params.real("strength", 1.0);
    if (!is_laplacian_kind(kind))
        throw std::invalid_argument("unknown Laplacian mask: " + kind);
    const Kernel kernel = laplacian_kernel(kind);
    const bool negative_centre = kernel[kernel.size() / 2][kernel[0].size() / 2] < 0.0;
    const double sign = negative_centre ? -1.0 : 1.0;

    std::vector<Plane> output;
    for (int c = 0; c < img.channels; ++c) {
        const Plane channel = channel_to_plane(img, c);
        const Plane lap = correlate(channel, kernel, Border::Reflect);
        Plane out(channel.width, channel.height);
        for (std::size_t i = 0; i < out.values.size(); ++i)
            out.values[i] = channel.values[i] + sign * strength * lap.values[i];
        output.push_back(out);
    }

    Result r;
    r.image = planes_to_image(output);
    r.aux.push_back({"Laplacian", plane_to_image(laplacian_abs(img, kind))});
    r.log = "g = f - c · ∇²f   (negative-centre mask)\nMask: " + kind + ", c = " +
            std::to_string(strength) +
            "\nSubtracting the Laplacian adds back the high-frequency detail, so\n"
            "edges become crisper.  The result is clipped to 0..255; large c can\n"
            "create halos.\n\n" +
            describe_kernel(kernel);
    return r;
}

}  // namespace

Operation op_laplacian_sharpening() {
    Operation op;
    op.key = "laplacian_sharp";
    op.name = "Laplacian sharpening";
    op.category = "Sharpening";
    op.description = "g = f - c·∇²f for a negative-centre mask, then clip to 0..255.";
    op.run = run;
    Param kind;
    kind.key = "kind";
    kind.label = "Mask";
    kind.kind = "choice";
    kind.def = "4-neighbor";
    kind.choices = {"4-neighbor", "8-neighbor", "diagonal"};
    Param strength;
    strength.key = "strength";
    strength.label = "Strength c";
    strength.kind = "float";
    strength.def = "1.0";
    strength.minimum = 0;
    strength.maximum = 4;
    strength.step = 0.1;
    op.params = {kind, strength};
    return op;
}

}  // namespace diplab
