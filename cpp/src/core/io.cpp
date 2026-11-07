#include "diplab/io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace diplab {
namespace {

std::string lower_extension(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

void skip_space(const std::vector<unsigned char>& buf, std::size_t& i) {
    while (i < buf.size()) {
        if (buf[i] == '#') {
            while (i < buf.size() && buf[i] != '\n') ++i;
        } else if (std::isspace(buf[i])) {
            ++i;
        } else {
            break;
        }
    }
}

bool read_token(const std::vector<unsigned char>& buf, std::size_t& i, std::string& out) {
    skip_space(buf, i);
    if (i >= buf.size()) return false;
    const std::size_t start = i;
    while (i < buf.size() && !std::isspace(buf[i]) && buf[i] != '#') ++i;
    out.assign(reinterpret_cast<const char*>(buf.data() + start), i - start);
    return !out.empty();
}

bool to_int(const std::string& text, int& value) {
    try {
        value = std::stoi(text);
        return true;
    } catch (...) {
        return false;
    }
}

std::uint32_t le32(const std::vector<unsigned char>& b, std::size_t at) {
    return static_cast<std::uint32_t>(b[at]) | (static_cast<std::uint32_t>(b[at + 1]) << 8) |
           (static_cast<std::uint32_t>(b[at + 2]) << 16) |
           (static_cast<std::uint32_t>(b[at + 3]) << 24);
}

std::uint16_t le16(const std::vector<unsigned char>& b, std::size_t at) {
    return static_cast<std::uint16_t>(b[at]) | (static_cast<std::uint16_t>(b[at + 1]) << 8);
}

void put_le32(std::vector<unsigned char>& b, std::uint32_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
}

void put_le16(std::vector<unsigned char>& b, std::uint16_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}

bool read_pnm(const std::vector<unsigned char>& buf, Image& out, std::string& error) {
    std::size_t i = 0;
    std::string magic;
    if (!read_token(buf, i, magic)) {
        error = "empty file";
        return false;
    }
    const bool gray = (magic == "P5" || magic == "P2");
    const bool colour = (magic == "P6" || magic == "P3");
    if (!gray && !colour) {
        error = "unsupported PNM magic (expected P5/P6, got " + magic + ")";
        return false;
    }
    const bool ascii = (magic == "P2" || magic == "P3");

    std::string tok;
    int width = 0, height = 0, maxval = 0;
    if (!read_token(buf, i, tok) || !to_int(tok, width) || !read_token(buf, i, tok) ||
        !to_int(tok, height) || !read_token(buf, i, tok) || !to_int(tok, maxval)) {
        error = "bad PNM header";
        return false;
    }
    if (width <= 0 || height <= 0 || maxval <= 0) {
        error = "bad PNM dimensions";
        return false;
    }
    if (maxval > 65535) {
        error = "PNM maxval too large";
        return false;
    }

    Image img(width, height, colour ? 3 : 1);
    const double scale = 255.0 / maxval;

    if (ascii) {
        for (std::size_t k = 0; k < img.values.size(); ++k) {
            int v = 0;
            if (!read_token(buf, i, tok) || !to_int(tok, v)) {
                error = "bad or truncated ASCII PNM data";
                return false;
            }
            img.values[k] = clamp_u8(v * scale);
        }
        out = std::move(img);
        return true;
    }

    if (i >= buf.size()) {
        error = "missing PNM pixel data";
        return false;
    }
    ++i;  // single whitespace after maxval
    const int bytes_per_sample = (maxval > 255) ? 2 : 1;
    const std::size_t need = img.values.size() * static_cast<std::size_t>(bytes_per_sample);
    if (buf.size() - i < need) {
        error = "truncated PNM pixel data";
        return false;
    }
    for (std::size_t k = 0; k < img.values.size(); ++k) {
        int v = buf[i];
        if (bytes_per_sample == 2) v = (buf[i] << 8) | buf[i + 1];
        i += static_cast<std::size_t>(bytes_per_sample);
        img.values[k] = clamp_u8(v * scale);
    }
    out = std::move(img);
    return true;
}

bool read_bmp(const std::vector<unsigned char>& buf, Image& out, std::string& error) {
    if (buf.size() < 54 || buf[0] != 'B' || buf[1] != 'M') {
        error = "not a BMP file";
        return false;
    }
    const std::uint32_t offset = le32(buf, 10);
    const std::uint32_t header_size = le32(buf, 14);
    if (header_size < 40) {
        error = "unsupported BMP header";
        return false;
    }
    const std::int32_t width = static_cast<std::int32_t>(le32(buf, 18));
    const std::int32_t raw_height = static_cast<std::int32_t>(le32(buf, 22));
    const std::uint16_t bpp = le16(buf, 28);
    const std::uint32_t compression = le32(buf, 30);
    if (bpp != 24 || compression != 0) {
        error = "only uncompressed 24-bit BMP is supported";
        return false;
    }
    if (width <= 0 || raw_height == 0) {
        error = "bad BMP dimensions";
        return false;
    }
    const bool top_down = raw_height < 0;
    const int height = top_down ? -raw_height : raw_height;
    const std::size_t stride = ((static_cast<std::size_t>(width) * 3 + 3) / 4) * 4;
    if (buf.size() < offset + stride * static_cast<std::size_t>(height)) {
        error = "truncated BMP data";
        return false;
    }

    Image img(width, height, 3);
    for (int y = 0; y < height; ++y) {
        const int src_row = top_down ? y : height - 1 - y;
        const std::size_t row = offset + stride * static_cast<std::size_t>(src_row);
        for (int x = 0; x < width; ++x) {
            const std::size_t px = row + static_cast<std::size_t>(x) * 3;
            img.at(y, x, 0) = buf[px + 2];  // BMP stores BGR
            img.at(y, x, 1) = buf[px + 1];
            img.at(y, x, 2) = buf[px + 0];
        }
    }
    out = std::move(img);
    return true;
}

bool write_pnm(const std::string& path, const Image& img, std::string& error) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open " + path + " for writing";
        return false;
    }
    f << (img.channels == 1 ? "P5" : "P6") << "\n"
      << "# diplab\n"
      << img.width << " " << img.height << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.values.data()),
            static_cast<std::streamsize>(img.values.size()));
    if (!f) {
        error = "failed while writing " + path;
        return false;
    }
    return true;
}

bool write_bmp(const std::string& path, const Image& img, std::string& error) {
    const Image rgb = to_rgb_image(img);
    const std::size_t stride = ((static_cast<std::size_t>(rgb.width) * 3 + 3) / 4) * 4;
    const std::size_t pixel_bytes = stride * static_cast<std::size_t>(rgb.height);
    const std::size_t file_size = 54 + pixel_bytes;

    std::vector<unsigned char> header;
    header.reserve(54);
    header.push_back('B');
    header.push_back('M');
    put_le32(header, static_cast<std::uint32_t>(file_size));
    put_le16(header, 0);
    put_le16(header, 0);
    put_le32(header, 54);
    put_le32(header, 40);
    put_le32(header, static_cast<std::uint32_t>(rgb.width));
    put_le32(header, static_cast<std::uint32_t>(rgb.height));
    put_le16(header, 1);
    put_le16(header, 24);
    put_le32(header, 0);
    put_le32(header, static_cast<std::uint32_t>(pixel_bytes));
    put_le32(header, 2835);
    put_le32(header, 2835);
    put_le32(header, 0);
    put_le32(header, 0);

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open " + path + " for writing";
        return false;
    }
    f.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));
    std::vector<unsigned char> row(stride, 0);
    for (int y = rgb.height - 1; y >= 0; --y) {
        std::fill(row.begin(), row.end(), 0);
        for (int x = 0; x < rgb.width; ++x) {
            const std::size_t px = static_cast<std::size_t>(x) * 3;
            row[px + 0] = rgb.at(y, x, 2);
            row[px + 1] = rgb.at(y, x, 1);
            row[px + 2] = rgb.at(y, x, 0);
        }
        f.write(reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(row.size()));
    }
    if (!f) {
        error = "failed while writing " + path;
        return false;
    }
    return true;
}

}  // namespace

bool read_image(const std::string& path, Image& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open " + path;
        return false;
    }
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    if (buf.empty()) {
        error = "empty file: " + path;
        return false;
    }
    const std::string ext = lower_extension(path);
    if (ext == "bmp") return read_bmp(buf, out, error);
    if (ext == "pgm" || ext == "ppm" || ext == "pnm" || ext == "pbm")
        return read_pnm(buf, out, error);
    // Unknown extension: sniff the magic bytes.
    if (buf.size() > 1 && buf[0] == 'B' && buf[1] == 'M') return read_bmp(buf, out, error);
    return read_pnm(buf, out, error);
}

bool write_image(const std::string& path, const Image& img, std::string& error) {
    if (img.empty()) {
        error = "refusing to write an empty image";
        return false;
    }
    if (lower_extension(path) == "bmp") return write_bmp(path, img, error);
    return write_pnm(path, img, error);
}

}  // namespace diplab
