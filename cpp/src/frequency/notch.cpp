// Zero two symmetric spikes in the spectrum.
#include "diplab/frequency_common.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double du = params.real("du", 40.0);
    const double dv = params.real("dv", 40.0);
    const double radius = params.real("radius", 6.0);
    const double pad = pad_factor_from(params);

    auto mask_fn = [&](int w, int h) { return notch_mask(w, h, du, dv, radius); };
    const std::string log = "Notch reject around (du, dv) = (" + std::to_string(du) + ", " +
                            std::to_string(dv) + ") and its mirror, radius = " +
                            std::to_string(radius) +
                            ".\nPeriodic noise appears as isolated bright spikes; zeroing\n"
                            "both symmetric spikes removes the pattern without blurring\n"
                            "the rest of the image.";
    return frequency_result(img, mask_fn, pad, log);
}

}  // namespace

Operation op_notch() {
    Operation op;
    op.key = "notch";
    op.name = "Notch reject (spikes)";
    op.category = "Frequency Domain";
    op.description = "Zero two symmetric spikes in the spectrum.";
    op.run = run;
    Param du;
    du.key = "du";
    du.label = "Offset du";
    du.kind = "float";
    du.def = "40";
    du.minimum = -400;
    du.maximum = 400;
    Param dv;
    dv.key = "dv";
    dv.label = "Offset dv";
    dv.kind = "float";
    dv.def = "40";
    dv.minimum = -400;
    dv.maximum = 400;
    Param radius;
    radius.key = "radius";
    radius.label = "Radius";
    radius.kind = "float";
    radius.def = "6";
    radius.minimum = 1;
    radius.maximum = 80;
    Param pad;
    pad.key = "padding";
    pad.label = "FFT padding";
    pad.kind = "choice";
    pad.def = "1";
    pad.choices = {"1", "2"};
    op.params = {du, dv, radius, pad};
    return op;
}

}  // namespace diplab
