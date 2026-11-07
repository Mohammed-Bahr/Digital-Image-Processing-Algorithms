// One colour plane of an RGB image.
#include "diplab/operations.hpp"

namespace diplab {
namespace {

int channel_index(const std::string& name) {
    if (name == "Green") return 1;
    if (name == "Blue") return 2;
    return 0;
}

Result run(const Image& img, const Params& params) {
    const std::string name = params.text("channel", "Red");
    const int index = channel_index(name);
    Result r;
    if (img.is_gray()) {
        r.image = img;
        r.log = "Input is already grayscale; the plane is returned unchanged.";
        return r;
    }
    r.image = plane_to_image(channel_to_plane(img, index));
    r.log = "A colour image is three stacked intensity matrices.\nKept channel " +
            std::to_string(index) + " (" + name + ").";
    return r;
}

}  // namespace

Operation op_extract_channel() {
    Operation op;
    op.key = "channel";
    op.name = "Extract colour channel";
    op.category = "Foundations";
    op.description = "Split the image into one of its three intensity planes.";
    op.run = run;
    Param p;
    p.key = "channel";
    p.label = "Channel";
    p.kind = "choice";
    p.def = "Red";
    p.choices = {"Red", "Green", "Blue"};
    op.params = {p};
    return op;
}

}  // namespace diplab
