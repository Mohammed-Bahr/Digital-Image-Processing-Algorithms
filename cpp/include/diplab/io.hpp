// Image IO without third-party libraries: PNM (PGM/PPM) and 24-bit BMP.
#pragma once

#include <string>

#include "diplab/image.hpp"

namespace diplab {

// Chooses the format from the file extension (.pgm/.ppm/.pnm/.bmp).
bool read_image(const std::string& path, Image& out, std::string& error);
bool write_image(const std::string& path, const Image& img, std::string& error);

}  // namespace diplab
