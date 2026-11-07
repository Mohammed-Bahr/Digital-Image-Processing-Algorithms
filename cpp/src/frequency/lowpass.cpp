// Keep the centre of the spectrum.
#include "diplab/frequency_common.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const MaskKind kind = kind_from(params);
    const std::string kind_name = params.text("kind", "butterworth");
    const double d0 = params.real("d0", 30.0);
    const double order = params.real("order", 2.0);
    const double pad = pad_factor_from(params);

    std::string formula;
    if (kind == MaskKind::Ideal)
        formula = "H = 1 if D(u,v) <= D0 else 0";
    else if (kind == MaskKind::Butterworth)
        formula = "H = 1 / (1 + (D/D0)^(2n))";
    else
        formula = "H = exp(-D^2 / 2D0^2)";

    auto mask_fn = [&](int w, int h) { return lowpass_mask(w, h, d0, order, kind); };
    const std::string log = kind_name +
                            " low-pass (smoothing / blurring)\n" + formula + "\nD0 = " +
                            std::to_string(d0) + ", padding factor = " + std::to_string(pad) +
                            "\nPipeline: f -> DFT -> multiply by H -> inverse DFT -> g.\n"
                            "The centre survives, the outside is attenuated.  Ideal filters\n"
                            "ring, Gaussian filters do not.";
    return frequency_result(img, mask_fn, pad, log);
}

}  // namespace

Operation op_lowpass() {
    Operation op;
    op.key = "lowpass";
    op.name = "Low-pass filter";
    op.category = "Frequency Domain";
    op.description = "Keep the centre, remove the outside: smooths the image.";
    op.run = run;
    Param kind;
    kind.key = "kind";
    kind.label = "Filter type";
    kind.kind = "choice";
    kind.def = "butterworth";
    kind.choices = {"ideal", "butterworth", "gaussian"};
    Param d0;
    d0.key = "d0";
    d0.label = "Cutoff D0";
    d0.kind = "float";
    d0.def = "30";
    d0.minimum = 1;
    d0.maximum = 400;
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
    op.params = {kind, d0, order, pad};
    return op;
}

}  // namespace diplab
