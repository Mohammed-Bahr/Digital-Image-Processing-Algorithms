// Shared plumbing for the frequency-domain operations.
#pragma once

#include <functional>
#include <string>

#include "diplab/fft.hpp"
#include "diplab/op.hpp"

namespace diplab {

double pad_factor_from(const Params& params);
MaskKind kind_from(const Params& params);

// Runs the mask, attaches spectra + mask images and reports how much of the
// spectrum survived.
Result frequency_result(const Image& img,
                        const std::function<Plane(int, int)>& mask_fn, double pad,
                        const std::string& log);

}  // namespace diplab
