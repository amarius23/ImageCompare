"""
Stage 1 — Smart Crop
Removes excess wall and floor so the furniture occupies ~50% of the frame.

Primary method : rembg background removal for a precise bounding box.
Fallback        : OpenCV edge-density analysis when rembg is unavailable.
"""

from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CropResult:
    image: np.ndarray
    x: int
    y: int
    w: int
    h: int


def crop_furniture(image_bgr: np.ndarray, floor_margin: float = 0.15) -> CropResult:
    """
    Returns a tight crop of the furniture with a small floor margin.
    floor_margin: fraction of furniture height to keep below the furniture bottom.
    """
    try:
        return _rembg_crop(image_bgr, floor_margin)
    except Exception as e:
        print(f"    [S1] rembg unavailable ({type(e).__name__}) — using edge fallback")
        return _edge_crop(image_bgr, floor_margin)


# ── shared helpers ────────────────────────────────────────────────────────────

def _long_line_map(image_bgr: np.ndarray) -> np.ndarray:
    """Returns a float32 map of long horizontal + vertical edge density."""
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 90)
    h, w    = image_bgr.shape[:2]
    min_len = max(30, int(min(h, w) * 0.05))
    h_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    v_kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kern)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kern)
    return cv2.add(h_lines, v_lines).astype(np.float32)


def _vertical_bounds(image_bgr: np.ndarray, x1: int, x2: int, h_img: int) -> tuple[int, int]:
    """
    Find furniture top/bottom within the x-strip.

    For the TOP bound we require that horizontal long-lines span at least
    40% of the strip width in that row.  This rejects narrow decorative items
    (picture frames, plants) sitting on top of the cabinet whose horizontal
    edges are much shorter than the cabinet's own top panel.

    For the BOTTOM bound a simple density threshold is sufficient.
    """
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges   = cv2.Canny(blurred, 30, 90)

    strip_w = max(x2 - x1, 1)
    min_len = max(30, int(strip_w * 0.05))

    # Horizontal long lines only (for top-detection)
    h_kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h_lines  = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kern)
    h_strip  = h_lines[:, x1:x2]                         # (h_img, strip_w)
    # coverage: fraction of strip width covered by horizontal lines per row
    h_cover  = h_strip.astype(np.float32).sum(axis=1) / (strip_w * 255.0)

    # Vertical long lines (for general density — y2)
    v_kern   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    v_lines  = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kern)

    # Combined density (smoothed) for y2
    combined = cv2.add(h_lines, v_lines).astype(np.float32)
    density  = cv2.GaussianBlur(combined, (71, 71), 0)
    col_den  = density[:, x1:x2].mean(axis=1)
    smooth   = np.convolve(col_den, np.ones(20) / 20, mode="same")
    thresh   = max(smooth.max() * 0.08, 1e-3)
    active   = np.where(smooth > thresh)[0]

    if len(active) == 0:
        return 0, h_img

    y2 = min(h_img, int(active[-1]) + 15)

    # For y1: use ONLY horizontal lines.
    # Plant stems and decorative verticals inflate the combined map above
    # the furniture, but the cabinet top panel is always a horizontal edge.
    h_thresh   = max(h_cover.max() * 0.08, 0.02)
    h_active   = np.where(h_cover > h_thresh)[0]
    if len(h_active) > 0:
        y1 = max(0, int(h_active[0]) - 5)
    else:
        y1 = max(0, int(active[0]) - 15)

    return y1, y2


# ── methods ───────────────────────────────────────────────────────────────────

def _rembg_crop(image_bgr: np.ndarray, floor_margin: float) -> CropResult:
    from rembg import remove as rembg_remove
    from PIL import Image as PILImage

    h_img, w_img = image_bgr.shape[:2]
    pil   = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    alpha = np.array(rembg_remove(pil))[:, :, 3]
    mask  = (alpha > 127).astype(np.uint8) * 255

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        raise RuntimeError("no foreground component found")

    cx_img = w_img / 2.0
    cy_img = h_img / 2.0
    diag   = (w_img ** 2 + h_img ** 2) ** 0.5

    def _score(lbl: int) -> float:
        s      = stats[lbl]
        cx     = s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]  / 2.0
        cy     = s[cv2.CC_STAT_TOP]  + s[cv2.CC_STAT_HEIGHT] / 2.0
        d      = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5 / diag
        # Furniture always sits on the floor — its bounding box bottom reaches
        # near the bottom of the image.  Wall art / pictures do not.
        # bottom_frac: fraction of image height that the bbox bottom reaches.
        # e.g. sideboard at 90% → 0.81,  wall picture at 50% → 0.25  (3× bias)
        bottom_frac = (s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]) / h_img
        return float(s[cv2.CC_STAT_AREA]) * (1.0 - d) ** 4 * (bottom_frac ** 2)

    best = max(range(1, n), key=_score)
    s    = stats[best]

    bbox_area = s[cv2.CC_STAT_WIDTH] * s[cv2.CC_STAT_HEIGHT]
    fill = s[cv2.CC_STAT_AREA] / bbox_area if bbox_area > 0 else 0
    if fill < 0.40:
        raise RuntimeError(f"best rembg component is not boxy (fill={fill:.2f}) — likely a lamp or plant")

    # Furniture sits on the floor — its bounding box must reach at least 60%
    # of the image height.  Wall-hung art, mirrors, and pictures do not.
    bottom_frac = (s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]) / h_img
    if bottom_frac < 0.60:
        raise RuntimeError(
            f"best rembg component ends at {bottom_frac:.0%} of image height "
            "— likely wall art, not floor furniture"
        )

    pad = 10
    x1  = max(0,     s[cv2.CC_STAT_LEFT] - pad)
    x2  = min(w_img, s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] + pad)

    # Derive vertical bounds from the rembg mask itself.
    # Row-wise mask coverage (fraction of strip width that is foreground)
    # is high where the cabinet body is (solid wide object) and low where
    # narrow items sit on top (plant stems, picture frame edges, vases).
    # → topmost row with coverage > 50% of max = actual furniture top edge.
    strip_mask   = mask[:, x1:x2].astype(np.float32) / 255.0
    row_coverage = strip_mask.mean(axis=1)           # shape (h_img,)
    max_cov      = row_coverage.max()

    furniture_rows = np.where(row_coverage > max_cov * 0.50)[0]
    if len(furniture_rows) == 0:
        furniture_rows = np.where(row_coverage > 0.01)[0]

    y1 = max(0,     int(furniture_rows[0])  - 5)
    y2 = min(h_img, int(furniture_rows[-1]) + 10)

    y2_floor = min(h_img, y2 + int((y2 - y1) * floor_margin))
    cropped  = image_bgr[y1:y2_floor, x1:x2]
    return CropResult(image=cropped, x=x1, y=y1, w=x2 - x1, h=y2_floor - y1)


def _edge_crop(image_bgr: np.ndarray, floor_margin: float) -> CropResult:
    h_img, w_img = image_bgr.shape[:2]

    # Pass 1 — horizontal: full-height column window to find furniture x-range
    line_map = _long_line_map(image_bgr)
    density  = cv2.GaussianBlur(line_map, (71, 71), 0)
    win_w    = max(1, int(w_img * 0.35))
    integral = cv2.integral(density)

    best_col_sum, best_x = -1.0, 0
    for x in range(0, w_img - win_w + 1, 8):
        col_sum = (integral[h_img, x + win_w] - integral[h_img, x]
                   - integral[0,     x + win_w] + integral[0,     x])
        if col_sum > best_col_sum:
            best_col_sum, best_x = col_sum, x

    # The sliding window finds the densest region but is capped at 35% of
    # the image width.  For wide furniture (sideboards, TV units) that spans
    # most of the image we expand the bounds outward: keep going left/right
    # while the column density stays above 15% of the mean within the seed window.
    col_sums = density.sum(axis=0)                       # per-column totals
    seed_mean = col_sums[best_x : best_x + win_w].mean()
    expand_thresh = seed_mean * 0.15
    active_cols = np.where(col_sums > expand_thresh)[0]
    if len(active_cols) >= 2:
        x1 = max(0,     int(active_cols[0])  - 15)
        x2 = min(w_img, int(active_cols[-1]) + 15)
    else:
        x1 = max(0,     best_x - 15)
        x2 = min(w_img, best_x + win_w + 15)

    # Pass 2 — vertical: row density within the x-strip
    y1, y2 = _vertical_bounds(image_bgr, x1, x2, h_img)

    y2_floor = min(h_img, y2 + int((y2 - y1) * floor_margin))
    cropped  = image_bgr[y1:y2_floor, x1:x2]
    return CropResult(image=cropped, x=x1, y=y1, w=x2 - x1, h=y2_floor - y1)
