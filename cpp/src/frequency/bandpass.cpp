// Keep a ring: low-pass(D1) * high-pass(D0).
#include "diplab/frequency_common.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const MaskKind kind = kind_from(params);
    const std::string kind_name = params.text("kind", "butterworth");
    const double d0 = params.real("d0", 15.0);
    const double d1 = params.real("d1", 55.0);
    const double order = params.real("order", 2.0);
    const double pad = pad_factor_from(params);

    auto mask_fn = [&](int w, int h) { return band_mask(w, h, d0, d1, order, kind, false); };
    const std::string log = kind_name + " band-pass: keep D0 <= D <= D1 (" +
                            std::to_string(d0) + " .. " + std::to_string(d1) +
                            ")\nBand-pass = low-pass(D1) · high-pass(D0).\n"
                            "It suppresses low frequencies (overall brightness) and high\n"
                            "frequencies (noise) at the same time.";
    return frequency_result(img, mask_fn, pad, log);
}

}  // namespace

Operation op_bandpass() {
    Operation op;
    op.key = "bandpass";
    op.name = "Band-pass filter";
    op.category = "Frequency Domain";
    op.description = "Keep only D0 <= D <= D1: isolates textures and periodic patterns.";
    op.run = run;
    Param kind;
    kind.key = "kind";
    kind.label = "Filter type";
    kind.kind = "choice";
    kind.def = "butterworth";
    kind.choices = {"ideal", "butterworth", "gaussian"};
    Param d0;
    d0.key = "d0";
    d0.label = "Lower D0";
    d0.kind = "float";
    d0.def = "15";
    d0.minimum = 1;
    d0.maximum = 400;
    Param d1;
    d1.key = "d1";
    d1.label = "Upper D1";
    d1.kind = "float";
    d1.def = "55";
    d1.minimum = 2;
    d1.maximum = 500;
    Param order;
    order.key = "order";
    order.label = "Order n";
    order.kind = "float";
    order.def = "2";
    order.minimum = 0.5;
    order.maximum = 10;
    order.step = 0.5;
    Param pad;
    pad.key = "padding";
    pad.label = "FFT padding";
    pad.kind = "choice";
    pad.def = "1";
    pad.choices = {"1", "2"};
    op.params = {kind, d0, d1, order, pad};
    return op;
}

}  // namespace diplab
