// f_hat = g - (sigma_eta^2 / sigma_L^2) * (g - m_L)
#include <algorithm>
#include <cmath>

#include "diplab/filters.hpp"
#include "diplab/operations.hpp"

namespace diplab {
namespace {

Result run(const Image& img, const Params& params) {
    const double noise_std = params.real("noise_std", 3.0);
    int k = params.integer("k", 3);
    if (k < 1) k = 1;
    const std::string border_name = params.text("border", "reflect");
    const Border border = parse_border(border_name);
    const double sigma2 = noise_std * noise_std;

    auto filter_plane = [&](const Plane& channel) {
        Plane squares(channel.width, channel.height);
        for (std::size_t i = 0; i < channel.values.size(); ++i)
            squares.values[i] = channel.values[i] * channel.values[i];

        const Plane mean = box_mean(channel, k, border);
        const Plane mean_squares = box_mean(squares, k, border);
        Plane out(channel.width, channel.height);
        for (std::size_t i = 0; i < out.values.size(); ++i) {
            const double variance = std::max(mean_squares.values[i] - mean.values[i] * mean.values[i], 0.0);
            const double ratio = std::min(sigma2 / std::max(variance, 1e-12), 1.0);
            out.values[i] = channel.values[i] - ratio * (channel.values[i] - mean.values[i]);
        }
        return out;
    };

    Result r;
    r.image = map_channels(img, filter_plane);
    r.log = "f̂ = g - (σ_η² / σ_L²) · (g - m_L)\nσ_η = " + std::to_string(noise_std) +
            ", window k = " + std::to_string(k) + ", border = " + border_name +
            "\nIn a flat region σ_L² is small, the ratio approaches 1 and the local\n"
            "mean is used.  On an edge σ_L² is large, the ratio approaches 0 and\n"
            "the original pixel is kept.";
    return r;
}

}  // namespace

Operation op_adaptive_local() {
    Operation op;
    op.key = "adaptive";
    op.name = "Adaptive local filter";
    op.category = "Smoothing";
    op.description = "f̂ = g - (σ_η²/σ_L²)(g - m_L): smooths flat areas, keeps edges.";
    op.run = run;
    Param noise;
    noise.key = "noise_std";
    noise.label = "Noise σ_η";
    noise.kind = "float";
    noise.def = "3.0";
    noise.minimum = 0;
    noise.maximum = 60;
    noise.step = 0.5;
    Param k;
    k.key = "k";
    k.label = "Window size";
    k.kind = "int";
    k.def = "3";
    k.minimum = 3;
    k.maximum = 15;
    k.step = 2;
    op.params = {noise, k, choice_border()};
    return op;
}

}  // namespace diplab
