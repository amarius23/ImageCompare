"""
Stage 4 — Edge Extraction
Produces a clean edge map that acts as a structural guide for the
wireframe generation step.

Output: white background with black lines representing:
  - Canny edges  (all structural boundaries)
  - Hough lines  (reinforced straight edges — furniture is mostly rectilinear)
"""

import cv2
import numpy as np


def extract_edges(image_bgr: np.ndarray, floor_margin: float = 0.15) -> np.ndarray:
    """
    Returns a BGR image (white background, black lines) at the same
    resolution as the input.

    floor_margin: fraction of image height at the bottom that is floor —
    edges there are blanked out to prevent carpet/floor noise from reaching
    the wireframe generation model.
    """
    h, w = image_bgr.shape[:2]
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    canny = cv2.Canny(blurred, 40, 120)

    # Blank the floor strip — it only contains noise
    floor_start = int(h * (1.0 - floor_margin))
    canny[floor_start:, :] = 0

    lines = cv2.HoughLinesP(
        canny,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=15,
        maxLineGap=8,
    )

    canvas = np.full((h, w), 255, dtype=np.uint8)
    canvas[canny > 0] = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Skip lines that fall in the floor strip
            if min(y1, y2) >= floor_start:
                continue
            cv2.line(canvas, (x1, y1), (x2, y2), 0, 1)

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
