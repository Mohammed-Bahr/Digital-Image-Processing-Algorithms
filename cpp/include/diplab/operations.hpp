// One factory per algorithm file. src/operations.cpp collects them.
#pragma once

#include "diplab/op.hpp"

namespace diplab {

// foundations
Operation op_grayscale_weighted();
Operation op_grayscale_average();
Operation op_extract_channel();

// point processing
Operation op_negative();
Operation op_logarithmic();
Operation op_gamma();
Operation op_piecewise_linear();
Operation op_contrast_stretching();
Operation op_histogram_equalization();

// noise
Operation op_salt_and_pepper();
Operation op_gaussian_noise();

// smoothing
Operation op_box_filter();
Operation op_weighted_average();
Operation op_median_filter();
Operation op_min_filter();
Operation op_max_filter();
Operation op_nth_order();
Operation op_adaptive_local();

// sharpening
Operation op_sobel();
Operation op_sobel_orientation();
Operation op_laplacian();
Operation op_laplacian_sharpening();

// frequency domain
Operation op_spectrum();
Operation op_lowpass();
Operation op_highpass();
Operation op_bandpass();
Operation op_bandreject();
Operation op_notch();
Operation op_homomorphic();

// features
Operation op_harris();
Operation op_harris_response();
Operation op_hog();

}  // namespace diplab
