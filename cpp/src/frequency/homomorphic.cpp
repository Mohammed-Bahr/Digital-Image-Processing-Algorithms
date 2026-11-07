// log -> DFT -> H -> IDFT -> exp
#include <algorithm>
#include <cmath>
#include <vector>

#include "diplab/filters.hpp"
#include "diplab/frequency_common.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double gl = params.real("gamma_l", 0.5);
    const double gh = params.real("gamma_h", 2.0);
    const double d0 = std::max(params.real("d0", 30.0), kEps);
    const double pad = pad_factor_from(params);

    const auto size = padded_size(img.height, img.width, pad);
    const int p = size.first;
    const int q = size.second;
    const Plane distance = distance_grid(q, p);

    Plane h(q, p);
    for (std::size_t i = 0; i < h.values.size(); ++i) {
        const double d2 = distance.values[i] * distance.values[i];
        h.values[i] = (gh - gl) * (1.0 - std::exp(-d2 / (2.0 * d0 * d0))) + gl;
    }

    std::vector<Plane> planes;
    for (int c = 0; c < img.channels; ++c) {
        const Plane channel = channel_to_plane(img, c);
        std::vector<Complex> f(static_cast<std::size_t>(p) * q, Complex(0.0, 0.0));
        for (int y = 0; y < channel.height; ++y)
            for (int x = 0; x < channel.width; ++x)
                f[static_cast<std::size_t>(y) * q + x] =
                    std::log1p(std::max(channel.at(y, x), 0.0));
        fft_2d(f, q, p, false);
        fft_shift_2d(f, q, p, false);
        for (std::size_t i = 0; i < f.size(); ++i) f[i] *= h.values[i];
        fft_shift_2d(f, q, p, true);
        fft_2d(f, q, p, true);

        Plane out(channel.width, channel.height);
        for (int y = 0; y < channel.height; ++y)
            for (int x = 0; x < channel.width; ++x) {
                const double s = f[static_cast<std::size_t>(y) * q + x].real();
                out.at(y, x) = std::expm1(std::min(std::max(s, -20.0), 20.0));
            }
        planes.push_back(out);
    }

    std::vector<double> all;
    for (const Plane& plane : planes)
        all.insert(all.end(), plane.values.begin(), plane.values.end());
    double lo = percentile_of(all, 2.0);
    double hi = percentile_of(all, 98.0);
    if (hi <= lo) {
        lo = *std::min_element(all.begin(), all.end());
        hi = *std::max_element(all.begin(), all.end());
    }
    if (hi > lo) {
        for (Plane& plane : planes)
            for (double& v : plane.values)
                v = std::min(std::max((v - lo) / (hi - lo) * 255.0, 0.0), 255.0);
    }

    int padded_w = 0;
    int padded_h = 0;
    const Plane spectrum = image_spectrum(img, pad, padded_w, padded_h);
    Plane h_display = h;
    for (double& v : h_display.values) v *= 255.0;

    Result r;
    r.image = planes_to_image(planes);
    r.aux.push_back({"Input spectrum", plane_to_image(spectrum)});
    r.aux.push_back({"Filter H", plane_to_image(h_display)});
    r.log = "f = i · r  ->  log  ->  DFT  ->  H  ->  IDFT  ->  exp  ->  g\nγ_L = " +
            std::to_string(gl) + ", γ_H = " + std::to_string(gh) + ", D0 = " +
            std::to_string(d0) +
            "\nH = (γ_H - γ_L)(1 - exp(-D²/2D0²)) + γ_L.\n"
            "The log separates illumination (low frequency) from reflectance (high\n"
            "frequency), so one filter can compress the first and boost the second.\n"
            "The output is rescaled (2nd..98th percentile) for display.";
    return r;
}

}  // namespace

Operation op_homomorphic() {
    Operation op;
    op.key = "homomorphic";
    op.name = "Homomorphic filtering";
    op.category = "Frequency Domain";
    op.description = "Compress illumination and boost reflectance at once.";
    op.run = run;
    Param gl;
    gl.key = "gamma_l";
    gl.label = "γ_L (illumination)";
    gl.kind = "float";
    gl.def = "0.5";
    gl.minimum = 0.05;
    gl.maximum = 1.0;
    gl.step = 0.05;
    Param gh;
    gh.key = "gamma_h";
    gh.label = "γ_H (reflectance)";
    gh.kind = "float";
    gh.def = "2.0";
    gh.minimum = 1.0;
    gh.maximum = 4.0;
    gh.step = 0.1;
    Param d0;
    d0.key = "d0";
    d0.label = "Cutoff D0";
    d0.kind = "float";
    d0.def = "30";
    d0.minimum = 1;
    d0.maximum = 400;
    Param pad;
    pad.key = "padding";
    pad.label = "FFT padding";
    pad.kind = "choice";
    pad.def = "1";
    pad.choices = {"1", "2"};
    op.params = {gl, gh, d0, pad};
    return op;
}

}  // namespace diplab
