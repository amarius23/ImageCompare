"""
Stage 1 — Furniture Localization
---------------------------------
Uses rembg (U2-Net) to remove the background, finds the largest foreground
object (the furniture), and returns a tightly cropped image plus its BBox.

Falls back to the full image if rembg is unavailable or fails.
"""

import cv2
import numpy as np

from .models import BBox


class FurnitureLocalizer:
    def __init__(self, padding: int = 20):
        self.padding = padding

    def run(self, image_bgr: np.ndarray) -> tuple[np.ndarray, BBox]:
        """
        Returns (cropped_image, bbox_in_original).
        """
        h, w = image_bgr.shape[:2]
        full_bbox = BBox(0, 0, w, h)

        try:
            mask = self._alpha_mask(image_bgr)
        except Exception as e:
            print(f"    [S1] rembg unavailable ({e}) — using full image")
            return image_bgr, full_bbox

        bbox = self._largest_component_bbox(mask, w, h)
        if bbox is None:
            return image_bgr, full_bbox

        x1 = max(0, bbox.x - self.padding)
        y1 = max(0, bbox.y - self.padding)
        x2 = min(w, bbox.x2 + self.padding)
        y2 = min(h, bbox.y2 + self.padding)

        cropped = image_bgr[y1:y2, x1:x2]
        return cropped, BBox(x1, y1, x2 - x1, y2 - y1)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _alpha_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        from rembg import remove as rembg_remove
        from PIL import Image as PILImage

        pil = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        removed = rembg_remove(pil)
        alpha = np.array(removed)[:, :, 3]
        return (alpha > 127).astype(np.uint8) * 255

    def _largest_component_bbox(self, mask: np.ndarray, img_w: int, img_h: int) -> BBox | None:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels < 2:
            return None

        # Score each component: area × centrality (prefer centred, large components)
        cx_img = img_w / 2.0
        cy_img = img_h / 2.0
        diag = (img_w ** 2 + img_h ** 2) ** 0.5
        best_lbl, best_score = 1, -1.0

        for lbl in range(1, n_labels):
            s = stats[lbl]
            cx = s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]  / 2.0
            cy = s[cv2.CC_STAT_TOP]  + s[cv2.CC_STAT_HEIGHT] / 2.0
            dist_frac = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5 / diag
            score = float(s[cv2.CC_STAT_AREA]) * (1.0 - dist_frac) ** 3
            if score > best_score:
                best_score, best_lbl = score, lbl

        s = stats[best_lbl]
        return BBox(
            s[cv2.CC_STAT_LEFT],
            s[cv2.CC_STAT_TOP],
            s[cv2.CC_STAT_WIDTH],
            s[cv2.CC_STAT_HEIGHT],
        )
