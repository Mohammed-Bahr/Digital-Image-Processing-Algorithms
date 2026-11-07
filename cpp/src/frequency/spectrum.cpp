// Centred log-magnitude spectrum.
#include "diplab/frequency_common.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double pad = pad_factor_from(params);
    int padded_w = 0;
    int padded_h = 0;
    const Plane spec = image_spectrum(img, pad, padded_w, padded_h);

    Result r;
    r.image = plane_to_image(spec);
    r.meta["padded_width"] = padded_w;
    r.meta["padded_height"] = padded_h;
    r.log = "Centred Fourier spectrum, displayed as log(1 + |F|).\nImage " +
            std::to_string(img.width) + "x" + std::to_string(img.height) +
            " zero-padded to " + std::to_string(padded_w) + "x" + std::to_string(padded_h) +
            ".\nCentre = DC (average brightness).  Near the centre = low frequencies,\n"
            "far from the centre = edges, detail and noise.";
    return r;
}

}  // namespace

Operation op_spectrum() {
    Operation op;
    op.key = "spectrum";
    op.name = "Fourier spectrum";
    op.category = "Frequency Domain";
    op.description = "Centred log-magnitude spectrum: DC in the middle, detail outside.";
    op.run = run;
    Param pad;
    pad.key = "padding";
    pad.label = "FFT padding";
    pad.kind = "choice";
    pad.def = "1";
    pad.choices = {"1", "2"};
    op.params = {pad};
    return op;
}

}  // namespace diplab
