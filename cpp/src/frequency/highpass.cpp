// Remove the centre of the spectrum.
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
    const bool rescale = params.flag("rescale", true);

    std::string formula;
    if (kind == MaskKind::Ideal)
        formula = "H = 0 if D(u,v) <= D0 else 1";
    else if (kind == MaskKind::Butterworth)
        formula = "H = 1 / (1 + (D0/D)^(2n))";
    else
        formula = "H = 1 - exp(-D^2 / 2D0^2)";

    auto mask_fn = [&](int w, int h) { return highpass_mask(w, h, d0, order, kind); };
    Result r = frequency_result(img, mask_fn, pad,
                                kind_name + " high-pass (sharpening / edges)\n" + formula +
                                    "\nD0 = " + std::to_string(d0) + ", padding factor = " +
                                    std::to_string(pad) +
                                    "\nThe DC term is zeroed, so smooth regions go black and\n"
                                    "only outlines remain.");

    if (rescale && r.image.maximum() > 0) {
        const int mn = r.image.minimum();
        const int mx = r.image.maximum();
        for (std::uint8_t& v : r.image.values) {
            if (mx > mn)
                v = clamp_u8((static_cast<double>(v) - mn) * 255.0 / (mx - mn));
            else
                v = 128;
        }
        r.log += "\nDisplay: output rescaled to the full 0..255 range (high-pass is dim).";
    }
    return r;
}

}  // namespace

Operation op_highpass() {
    Operation op;
    op.key = "highpass";
    op.name = "High-pass filter";
    op.category = "Frequency Domain";
    op.description = "Remove the centre, keep edges. The DC term is zeroed.";
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
    Param rescale;
    rescale.key = "rescale";
    rescale.label = "Rescale for display";
    rescale.kind = "bool";
    rescale.def = "true";
    Param pad;
    pad.key = "padding";
    pad.label = "FFT padding";
    pad.kind = "choice";
    pad.def = "1";
    pad.choices = {"1", "2"};
    op.params = {kind, d0, order, rescale, pad};
    return op;
}

}  // namespace diplab
