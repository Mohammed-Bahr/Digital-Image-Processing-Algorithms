// Radix-2 FFT, spectrum helpers and frequency-domain masks.
#pragma once

#include <complex>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "diplab/image.hpp"

namespace diplab {

using Complex = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;

int next_power_of_two(int n);

// In-place, length must be a power of two.
void fft_1d(std::vector<Complex>& a, bool inverse);
// In-place on a width*height buffer, both dimensions powers of two.
void fft_2d(std::vector<Complex>& a, int width, int height, bool inverse);

// Moves the origin to the centre (and back when inverse_shift is true).
void fft_shift_2d(std::vector<Complex>& a, int width, int height,
                  bool inverse_shift);

// D(u,v) measured from the centred origin.
Plane distance_grid(int width, int height);

enum class MaskKind { Ideal, Butterworth, Gaussian };
MaskKind parse_mask_kind(const std::string& name);
bool mask_kind_valid(const std::string& name);

Plane lowpass_mask(int width, int height, double d0, double order, MaskKind kind);
Plane highpass_mask(int width, int height, double d0, double order, MaskKind kind);
Plane band_mask(int width, int height, double d0, double d1, double order,
                MaskKind kind, bool reject);
Plane notch_mask(int width, int height, double du, double dv, double radius);

std::pair<int, int> padded_size(int height, int width, double factor);

struct FilterOutput {
    Image image;
    Plane mask;
    Plane spectrum_input;
    Plane spectrum_output;
    int padded_width = 0;
    int padded_height = 0;
};

// f -> DFT -> shift -> multiply by H -> unshift -> IDFT -> real part.
// mask_fn(width, height) builds H on the padded grid.
FilterOutput apply_frequency_filter(
    const Image& img, const std::function<Plane(int, int)>& mask_fn,
    double pad_factor = 1.0);

// (log(1 + |F|) / peak)^gamma scaled to 0..255.
Plane spectrum_display(const std::vector<Complex>& centred, int width, int height,
                       double gamma = 1.0);

// Log-magnitude spectrum of the image's luminance plane.
Plane image_spectrum(const Image& img, double pad_factor, int& padded_w,
                     int& padded_h);

}  // namespace diplab
