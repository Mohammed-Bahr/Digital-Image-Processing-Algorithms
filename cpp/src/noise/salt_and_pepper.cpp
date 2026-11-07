// Half the selected pixels become 0 (pepper), half become 255 (salt).
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "diplab/operations.hpp"

namespace diplab {
namespace {

Image add_salt_and_pepper(const Image& img, double amount, int seed) {
    Image out = img;
    const double pct = std::min(std::max(amount, 0.0), 100.0);
    const std::size_t total = out.values.size();
    const std::size_t n = static_cast<std::size_t>(std::llround(total * pct / 100.0));
    if (n == 0) return out;

    std::vector<std::size_t> index(total);
    std::iota(index.begin(), index.end(), std::size_t{0});
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    const std::size_t take = std::min(n, total);
    std::shuffle(index.begin(), index.end(), rng);

    const std::size_t half = take / 2;
    for (std::size_t i = 0; i < half; ++i) out.values[index[i]] = 0;
    for (std::size_t i = half; i < take; ++i) out.values[index[i]] = 255;
    return out;
}

Result run(const Image& img, const Params& params) {
    const double amount = params.real("amount", 3.0);
    const int seed = params.integer("seed", 0);

    Result r;
    r.image = add_salt_and_pepper(img, amount, seed);
    r.log = "Salt & pepper noise on " + std::to_string(amount) + "% of the pixels (seed " +
            std::to_string(seed) +
            ").\nHalf become 0 (pepper), half become 255 (salt).  Impulse noise\n"
            "is best removed by a median, not a linear average.";
    return r;
}

}  // namespace

Operation op_salt_and_pepper() {
    Operation op;
    op.key = "salt";
    op.name = "Add salt & pepper noise";
    op.category = "Noise";
    op.description = "Random pixels are forced to 0 (pepper) or 255 (salt).";
    op.run = run;
    Param amount;
    amount.key = "amount";
    amount.label = "Noise %";
    amount.kind = "float";
    amount.def = "3";
    amount.minimum = 0;
    amount.maximum = 30;
    amount.step = 0.5;
    Param seed;
    seed.key = "seed";
    seed.label = "Seed";
    seed.kind = "int";
    seed.def = "0";
    seed.minimum = 0;
    seed.maximum = 9999;
    op.params = {amount, seed};
    return op;
}

}  // namespace diplab
