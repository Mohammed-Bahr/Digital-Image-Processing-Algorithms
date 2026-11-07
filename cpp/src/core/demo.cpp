#include "diplab/demo.hpp"

#include <algorithm>
#include <cmath>

namespace diplab {
namespace {
constexpr double kDemoPi = 3.14159265358979323846;
}  // namespace

Image make_demo_image(int size) {
    const int w = size;
    const int h = size;
    Image img(w, h, 3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // smooth, uneven illumination
            const double base = 118.0 + 52.0 * (1.0 - static_cast<double>(x) / w) +
                                26.0 * std::cos(static_cast<double>(y) / h * kDemoPi);
            double r = base * 0.94;
            double g = base * 0.99;
            double b = base * 1.08;

            // large shadow, bottom-left
            if (x < 150 && y > 250) {
                r *= 0.38;
                g *= 0.38;
                b *= 0.38;
            }

            // sun and its halo
            const double sdx = x - 105.0;
            const double sdy = y - 96.0;
            const double sd2 = sdx * sdx + sdy * sdy;
            if (sd2 <= 58.0 * 58.0) {
                r = 250.0;
                g = 205.0;
                b = 90.0;
            } else if (sd2 <= 76.0 * 76.0) {
                r = r * 0.35 + 250.0 * 0.65;
                g = g * 0.35 + 205.0 * 0.65;
                b = b * 0.35 + 90.0 * 0.65;
            }

            // buildings
            if (y >= 150 && y < 470 && x >= 176 && x < 332) {
                r = 58;
                g = 66;
                b = 96;
            }
            if (y >= 236 && y < 470 && x >= 344 && x < 468) {
                r = 84;
                g = 74;
                b = 98;
            }

            // lit windows
            for (int row = 0; row < 6; ++row) {
                const int y0 = 176 + row * 44;
                if (y < y0 || y >= y0 + 22) continue;
                for (int col = 0; col < 3; ++col) {
                    const int x0 = 196 + col * 44;
                    if (x >= x0 && x < x0 + 26) {
                        r = 252;
                        g = 222;
                        b = 140;
                    }
                }
            }
            for (int row = 0; row < 4; ++row) {
                const int y0 = 270 + row * 46;
                if (y < y0 || y >= y0 + 18) continue;
                for (int col = 0; col < 2; ++col) {
                    const int x0 = 366 + col * 44;
                    if (x >= x0 && x < x0 + 24) {
                        r = 170;
                        g = 215;
                        b = 245;
                    }
                }
            }

            // ground
            if (y >= 470) {
                r = 78;
                g = 112;
                b = 84;
                if (y < 474) {
                    r = 48;
                    g = 70;
                    b = 52;
                }
            }

            // checkerboard texture
            if (y >= 40 && y < 168 && x >= 372 && x < 500) {
                const bool light = ((x / 11) + (y / 11)) % 2 == 1;
                if (light) {
                    r = 232;
                    g = 232;
                    b = 226;
                } else {
                    r = 34;
                    g = 36;
                    b = 40;
                }
            }

            // red ball with a specular highlight
            const double bdx = x - 424.0;
            const double bdy = y - 388.0;
            if (bdx * bdx + bdy * bdy <= 42.0 * 42.0) {
                r = 196;
                g = 62;
                b = 58;
            }
            const double hdx = x - 410.0;
            const double hdy = y - 374.0;
            if (hdx * hdx + hdy * hdy <= 12.0 * 12.0) {
                r = 255;
                g = 196;
                b = 188;
            }

            // thin vertical structures
            for (int k = 0; k < 3; ++k) {
                const int x0 = 46 + k * 26;
                if (y >= 300 && y < 430 && x >= x0 && x < x0 + 4) {
                    r = 210;
                    g = 214;
                    b = 220;
                }
            }

            // thin diagonal structure
            if (std::fabs((y - 430) + (x - 30) * 0.55) < 2.2 && x > 20 && x < 175 &&
                y > 300 && y < 470) {
                r = 235;
                g = 235;
                b = 240;
            }

            // vignette
            const double dx = x - w / 2.0;
            const double dy = y - h / 2.0;
            const double r2 = (dx * dx + dy * dy) / (static_cast<double>(w) * w);
            const double v = 1.0 - 0.35 * r2;
            img.at(y, x, 0) = clamp_u8(r * v);
            img.at(y, x, 1) = clamp_u8(g * v);
            img.at(y, x, 2) = clamp_u8(b * v);
        }
    }
    return img;
}

}  // namespace diplab
