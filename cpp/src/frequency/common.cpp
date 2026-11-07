#include "diplab/frequency_common.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace diplab {

double pad_factor_from(const Params& params) {
    try {
        return std::max(1.0, std::stod(params.text("padding", "1")));
    } catch (...) {
        return 1.0;
    }
}

MaskKind kind_from(const Params& params) {
    const std::string name = params.text("kind", "butterworth");
    if (!mask_kind_valid(name)) throw std::invalid_argument("unknown filter type: " + name);
    return parse_mask_kind(name);
}

Result frequency_result(const Image& img,
                        const std::function<Plane(int, int)>& mask_fn, double pad,
                        const std::string& log) {
    const FilterOutput fo = apply_frequency_filter(img, mask_fn, pad);

    double kept = 0.0;
    for (double v : fo.mask.values) kept += v;
    if (!fo.mask.values.empty()) kept = 100.0 * kept / static_cast<double>(fo.mask.values.size());

    Plane mask_display = fo.mask;
    for (double& v : mask_display.values) v *= 255.0;

    Result r;
    r.image = fo.image;
    r.log = log + "\n\nMask mean = " + std::to_string(kept) + "% of the spectrum kept.";
    r.aux.push_back({"Input |F|", plane_to_image(fo.spectrum_input)});
    r.aux.push_back({"Filter H", plane_to_image(mask_display)});
    r.aux.push_back({"Output |G|", plane_to_image(fo.spectrum_output)});
    r.meta["padded_width"] = fo.padded_width;
    r.meta["padded_height"] = fo.padded_height;
    return r;
}

}  // namespace diplab
