#include "diplab/op.hpp"

#include <algorithm>
#include <stdexcept>

#include "diplab/operations.hpp"

namespace diplab {
namespace {

std::string lowercase(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

void Params::set(const std::string& key, const std::string& value) {
    values_[key] = value;
}

bool Params::has(const std::string& key) const {
    return values_.find(key) != values_.end();
}

std::string Params::text(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
}

int Params::integer(const std::string& key, int fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (...) {
        return fallback;
    }
}

double Params::real(const std::string& key, double fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    try {
        return std::stod(it->second);
    } catch (...) {
        return fallback;
    }
}

bool Params::flag(const std::string& key, bool fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const std::string v = lowercase(it->second);
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return fallback;
}

Param choice_border() {
    Param p;
    p.key = "border";
    p.label = "Border padding";
    p.kind = "choice";
    p.def = "reflect";
    p.choices = {"reflect", "replicate", "zeros"};
    return p;
}

Params default_params(const Operation& op) {
    Params params;
    for (const Param& p : op.params) params.set(p.key, p.def);
    return params;
}

Result apply(const Operation& op, const Image& img, const Params& user) {
    if (op.run == nullptr) throw std::runtime_error("operation " + op.key + " has no run function");
    if (img.empty()) throw std::runtime_error("input image is empty");

    Params params = default_params(op);
    for (const auto& kv : user.all()) params.set(kv.first, kv.second);
    return op.run(img, params);
}

const std::vector<std::string>& category_order() {
    static const std::vector<std::string> order = {
        "Foundations", "Point Processing", "Noise",   "Smoothing",
        "Sharpening",  "Frequency Domain", "Features"};
    return order;
}

const std::vector<Operation>& operations() {
    static const std::vector<Operation> all = {
        // foundations
        op_grayscale_weighted(), op_grayscale_average(), op_extract_channel(),
        // point processing
        op_negative(), op_logarithmic(), op_gamma(), op_piecewise_linear(),
        op_contrast_stretching(), op_histogram_equalization(),
        // noise
        op_salt_and_pepper(), op_gaussian_noise(),
        // smoothing
        op_box_filter(), op_weighted_average(), op_median_filter(), op_min_filter(),
        op_max_filter(), op_nth_order(), op_adaptive_local(),
        // sharpening
        op_sobel(), op_sobel_orientation(), op_laplacian(), op_laplacian_sharpening(),
        // frequency domain
        op_spectrum(), op_lowpass(), op_highpass(), op_bandpass(), op_bandreject(),
        op_notch(), op_homomorphic(),
        // features
        op_harris(), op_harris_response(), op_hog()};
    return all;
}

const Operation* find_operation(const std::string& key) {
    for (const Operation& op : operations())
        if (op.key == key) return &op;
    return nullptr;
}

}  // namespace diplab
