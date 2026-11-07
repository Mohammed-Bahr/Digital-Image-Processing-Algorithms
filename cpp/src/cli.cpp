// Command-line front end: list / run / demo / help.
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "diplab/demo.hpp"
#include "diplab/io.hpp"
#include "diplab/op.hpp"

namespace {

using namespace diplab;

void usage() {
    std::cout << "DIP Lab (C++) - digital image processing algorithms as pure math\n\n"
                 "usage:\n"
                 "  diplab list [--op KEY]          list algorithms and their parameters\n"
                 "  diplab run OP IN OUT [key=val]  run one algorithm\n"
                 "  diplab demo OUT                 write the synthetic demo scene\n"
                 "  diplab help                     this text\n\n"
                 "examples:\n"
                 "  diplab demo scene.ppm\n"
                 "  diplab run box scene.ppm out.ppm k=5 border=reflect\n"
                 "  diplab run lowpass scene.ppm out.bmp kind=gaussian d0=25\n\n"
                 "formats: .pgm/.ppm (PGM/PPM) and .bmp (24-bit) in and out.\n"
                 "add --log to print the operation's explanation, --save-aux to\n"
                 "also write the auxiliary images (spectra, filter masks, ...).\n";
}

std::string format_param(const Param& p) {
    std::string text = p.key + "=" + p.def;
    if (p.kind == "choice") {
        text += " [";
        for (std::size_t i = 0; i < p.choices.size(); ++i) {
            if (i) text += "|";
            text += p.choices[i];
        }
        text += "]";
    } else if (p.kind == "int" || p.kind == "float") {
        text += " (" + std::to_string(static_cast<long long>(p.minimum)) + ".." +
                std::to_string(static_cast<long long>(p.maximum)) + ")";
    }
    return text;
}

void print_operation(const Operation& op, bool verbose) {
    std::cout << "  " << op.key;
    for (std::size_t i = op.key.size(); i < 16; ++i) std::cout << ' ';
    std::cout << op.name << "\n";
    if (!verbose) return;
    std::cout << "      " << op.description << "\n      category: " << op.category << "\n";
    for (const Param& p : op.params)
        std::cout << "      --" << format_param(p) << "   " << p.label << "\n";
}

int command_list(const std::vector<std::string>& args) {
    std::string only;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--op" && i + 1 < args.size()) only = args[i + 1];
    }
    if (!only.empty()) {
        const Operation* op = find_operation(only);
        if (op == nullptr) {
            std::cerr << "unknown operation: " << only << "\n";
            return 1;
        }
        print_operation(*op, true);
        return 0;
    }

    for (const std::string& category : category_order()) {
        std::cout << "\n" << category << "\n";
        for (const Operation& op : operations())
            if (op.category == category) print_operation(op, false);
    }
    std::cout << "\n" << operations().size() << " operations. "
                 "Use 'diplab list --op KEY' for parameters.\n";
    return 0;
}

std::string strip_dashes(const std::string& text) {
    std::size_t start = 0;
    while (start + 1 < text.size() && text[start] == '-') ++start;
    return text.substr(start);
}

std::string stem_of(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    return dot == std::string::npos ? path : path.substr(0, dot);
}

int command_run(const std::vector<std::string>& args) {
    if (args.size() < 3) {
        std::cerr << "usage: diplab run OP IN OUT [key=val ...]\n";
        return 1;
    }
    const std::string key = args[0];
    const std::string in_path = args[1];
    const std::string out_path = args[2];

    const Operation* op = find_operation(key);
    if (op == nullptr) {
        std::cerr << "unknown operation: " << key << "\n";
        return 1;
    }

    Params params;
    bool print_log = false;
    bool save_aux = false;
    for (std::size_t i = 3; i < args.size(); ++i) {
        const std::string token = strip_dashes(args[i]);
        if (token == "log" || token == "v") {
            print_log = true;
            continue;
        }
        if (token == "save-aux") {
            save_aux = true;
            continue;
        }
        const std::size_t eq = token.find('=');
        if (eq == std::string::npos) {
            std::cerr << "expected key=value, got: " << args[i] << "\n";
            return 1;
        }
        params.set(token.substr(0, eq), token.substr(eq + 1));
    }

    Image input;
    std::string error;
    if (!read_image(in_path, input, error)) {
        std::cerr << "read failed: " << error << "\n";
        return 1;
    }

    Result result;
    try {
        result = apply(*op, input, params);
    } catch (const std::exception& e) {
        std::cerr << key << " failed: " << e.what() << "\n";
        return 1;
    }

    if (!write_image(out_path, result.image, error)) {
        std::cerr << "write failed: " << error << "\n";
        return 1;
    }
    std::cout << op->name << ": " << input.width << "x" << input.height << " -> "
              << result.image.width << "x" << result.image.height << " ("
              << result.image.channels << " channel" << (result.image.channels == 1 ? "" : "s")
              << ")  wrote " << out_path << "\n";
    if (print_log && !result.log.empty()) std::cout << "\n" << result.log << "\n";

    if (save_aux) {
        const std::string stem = stem_of(out_path);
        for (std::size_t i = 0; i < result.aux.size(); ++i) {
            const std::string aux_path = stem + "_aux" + std::to_string(i + 1) + ".ppm";
            if (write_image(aux_path, result.aux[i].second, error)) {
                std::cout << "  aux " << result.aux[i].first << " -> " << aux_path << "\n";
            } else {
                std::cerr << "  aux write failed: " << error << "\n";
            }
        }
    }
    return 0;
}

int command_demo(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "usage: diplab demo OUT\n";
        return 1;
    }
    const Image scene = make_demo_image(512);
    std::string error;
    if (!write_image(args[0], scene, error)) {
        std::cerr << "write failed: " << error << "\n";
        return 1;
    }
    std::cout << "wrote demo scene " << args[0] << " (" << scene.width << "x" << scene.height
              << ")\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string command = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);

    if (command == "list") return command_list(args);
    if (command == "run") return command_run(args);
    if (command == "demo") return command_demo(args);
    if (command == "help" || command == "--help" || command == "-h") {
        usage();
        return 0;
    }
    std::cerr << "unknown command: " << command << "\n\n";
    usage();
    return 1;
}
