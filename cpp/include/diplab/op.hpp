// One algorithm = one Operation: key, metadata, parameters and a run function.
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "diplab/image.hpp"

namespace diplab {

struct Param {
    std::string key;
    std::string label;
    std::string kind = "int";  // int | float | choice | bool
    std::string def = "0";     // default, as text
    double minimum = 0.0;
    double maximum = 100.0;
    double step = 1.0;
    std::vector<std::string> choices;
};

struct Result {
    Image image;
    std::string log;
    std::vector<std::pair<std::string, Image>> aux;
    std::map<std::string, double> meta;
};

// key -> value, all values kept as text (CLI friendly).
class Params {
  public:
    void set(const std::string& key, const std::string& value);
    bool has(const std::string& key) const;

    std::string text(const std::string& key, const std::string& fallback) const;
    int integer(const std::string& key, int fallback) const;
    double real(const std::string& key, double fallback) const;
    bool flag(const std::string& key, bool fallback) const;

    const std::map<std::string, std::string>& all() const { return values_; }

  private:
    std::map<std::string, std::string> values_;
};

struct Operation {
    std::string key;
    std::string name;
    std::string category;
    std::string description;
    std::vector<Param> params;
    Result (*run)(const Image&, const Params&) = nullptr;
};

Params default_params(const Operation& op);
// Fills in the operation's defaults, then applies the user's overrides.
Result apply(const Operation& op, const Image& img, const Params& user);

// Shared "border" parameter: reflect / replicate / zeros.
Param choice_border();

const std::vector<Operation>& operations();
const Operation* find_operation(const std::string& key);
const std::vector<std::string>& category_order();

}  // namespace diplab
