"""A synthetic scene so the app is useful even before you open a photo.

It intentionally contains: smooth illumination gradients, dark shadows, strong
edges, corners, fine texture (checkerboard), bright/dark regions and colour --
so every operator in the notes has something to act on.
"""

from __future__ import annotations

import numpy as np

from .utils import as_uint8


def make_demo_image(size: int = 512) -> np.ndarray:
    h = w = int(size)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # --- smooth, uneven illumination -------------------------------------
    base = 118 + 52 * (1.0 - xx / w) + 26 * np.cos(yy / h * np.pi)
    img = np.stack([base * 0.94, base * 0.99, base * 1.08], axis=2)

    # --- large shadow region (bottom-left) -------------------------------
    shadow = (xx < 150) & (yy > 250)
    img[shadow] = img[shadow] * 0.38

    # --- sun (very bright, saturated area) --------------------------------
    sun = (xx - 105) ** 2 + (yy - 96) ** 2 <= 58 ** 2
    img[sun] = (250, 205, 90)
    halo = ((xx - 105) ** 2 + (yy - 96) ** 2 <= 76 ** 2) & ~sun
    img[halo] = img[halo] * 0.35 + np.array([250, 205, 90]) * 0.65

    # --- buildings --------------------------------------------------------
    img[150:470, 176:332] = (58, 66, 96)
    img[236:470, 344:468] = (84, 74, 98)

    # lit windows (regular grid -> many corners)
    for row in range(6):
        for col in range(3):
            y0 = 176 + row * 44
            x0 = 196 + col * 44
            img[y0:y0 + 22, x0:x0 + 26] = (252, 222, 140)

    for row in range(4):
        for col in range(2):
            y0 = 270 + row * 46
            x0 = 366 + col * 44
            img[y0:y0 + 18, x0:x0 + 24] = (170, 215, 245)

    # --- ground ------------------------------------------------------------
    img[470:, :] = (78, 112, 84)
    img[470:474, :] = (48, 70, 52)

    # --- checkerboard texture (high frequency) ----------------------------
    check = ((xx // 11).astype(int) + (yy // 11).astype(int)) % 2
    patch = (slice(40, 168), slice(372, 500))
    sub = check[patch][..., None]
    img[patch] = np.where(sub == 1,
                          np.array([232, 232, 226]),
                          np.array([34, 36, 40]))

    # --- red ball + highlight (round object, specular detail) -------------
    ball = (xx - 424) ** 2 + (yy - 388) ** 2 <= 42 ** 2
    img[ball] = (196, 62, 58)
    hi = (xx - 410) ** 2 + (yy - 374) ** 2 <= 12 ** 2
    img[hi] = (255, 196, 188)

    # --- a few thin structures (edges at several orientations) ------------
    for k in range(3):
        x0 = 46 + k * 26
        img[300:430, x0:x0 + 4] = (210, 214, 220)
    diag = np.abs((yy - 430) + (xx - 30) * 0.55) < 2.2
    diag &= (xx > 20) & (xx < 175) & (yy > 300) & (yy < 470)
    img[diag] = (235, 235, 240)

    # --- vignette ----------------------------------------------------------
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * w)
    img = img * (1.0 - 0.35 * r2)[..., None]

    return as_uint8(img)
