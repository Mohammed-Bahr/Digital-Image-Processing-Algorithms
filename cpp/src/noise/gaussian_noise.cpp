// g = f + eta, eta ~ N(0, sigma^2)
#include <random>

#include "diplab/operations.hpp"

namespace diplab {
namespace {

Image add_gaussian_noise(const Image& img, double sigma, int seed) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<double> dist(0.0, std::max(sigma, 0.0));
    Image out = img;
    for (std::size_t i = 0; i < out.values.size(); ++i)
        out.values[i] = clamp_u8(static_cast<double>(out.values[i]) + dist(rng));
    return out;
}

Result run(const Image& img, const Params& params) {
    const double sigma = params.real("sigma", 15.0);
    const int seed = params.integer("seed", 0);

    Result r;
    r.image = add_gaussian_noise(img, sigma, seed);
    r.log = "g = f + η,   η ~ N(0, σ²),   σ = " + std::to_string(sigma) + " (seed " +
            std::to_string(seed) +
            ")\nA linear average reduces Gaussian noise but also blurs edges.";
    return r;
}

}  // namespace

Operation op_gaussian_noise() {
    Operation op;
    op.key = "gauss_noise";
    op.name = "Add Gaussian noise";
    op.category = "Noise";
    op.description = "g = f + η with η ~ N(0, σ²).";
    op.run = run;
    Param sigma;
    sigma.key = "sigma";
    sigma.label = "σ";
    sigma.kind = "float";
    sigma.def = "15";
    sigma.minimum = 0;
    sigma.maximum = 100;
    Param seed;
    seed.key = "seed";
    seed.label = "Seed";
    seed.kind = "int";
    seed.def = "0";
    seed.minimum = 0;
    seed.maximum = 9999;
    op.params = {sigma, seed};
    return op;
}

}  // namespace diplab
