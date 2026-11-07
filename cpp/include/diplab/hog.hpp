// Histogram of Oriented Gradients (unsigned gradients, soft binning, L2 blocks).
#pragma once

#include <vector>

#include "diplab/image.hpp"

namespace diplab {

struct HogDescriptor {
    std::vector<double> vector;   // the descriptor itself
    std::vector<double> hist;     // cells_y * cells_x * bins
    int cells_y = 0;
    int cells_x = 0;
    int blocks_y = 0;
    int blocks_x = 0;
    int cell = 8;
    int bins = 9;
    int block = 2;
    int width = 0;
    int height = 0;
    Plane magnitude;
    Plane angle;
};

HogDescriptor hog_compute(const Image& img, int cell = 8, int bins = 9,
                          int block = 2, int stride = 1,
                          bool resize_to_window = false);

// One orientation fan per cell, the classic HOG picture.
Image hog_visualization(const std::vector<double>& hist, int cells_y,
                        int cells_x, int bins, int scale);

}  // namespace diplab
