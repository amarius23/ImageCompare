import cv2
import numpy as np

from .models import WireframeResult

_CANNY_LOW   = 35
_CANNY_HIGH  = 110
_LINE_KERNEL = 3     # dilation kernel size
_NOISE_OPEN  = 3     # morphological open kernel
_SMOOTH_SIG  = 1.2   # Gaussian sigma for display wireframe
_SOFT_SIG    = 14    # Gaussian sigma for comparison soft map


class WireframeGenerator:
    def run(self, image_bgr: np.ndarray) -> WireframeResult:
        h, w = image_bgr.shape[:2]
        edges = _canny(image_bgr)
        # thicken lines
        k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (_LINE_KERNEL, _LINE_KERNEL))
        thick = cv2.dilate(edges, k_line, iterations=1)
        # remove isolated noise blobs
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (_NOISE_OPEN, _NOISE_OPEN))
        thick = cv2.morphologyEx(thick, cv2.MORPH_OPEN, k_open)
        # soft map for comparison
        soft_raw = cv2.GaussianBlur(thick.astype(np.float32), (0, 0), _SOFT_SIG, _SOFT_SIG)
        mx = soft_raw.max()
        soft_map = (soft_raw / mx).astype(np.float32) if mx > 0 else soft_raw
        # display wireframe: smooth + invert
        smoothed = cv2.GaussianBlur(thick.astype(np.float32), (0, 0), _SMOOTH_SIG, _SMOOTH_SIG)
        smoothed = (smoothed / smoothed.max() * 255).astype(np.uint8) if smoothed.max() > 0 else smoothed.astype(np.uint8)
        clean_img = cv2.bitwise_not(smoothed)
        return WireframeResult(clean_img=clean_img, soft_map=soft_map,
                               aspect_ratio=round(w / h, 4) if h > 0 else 1.0)


def _canny(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, _CANNY_LOW, _CANNY_HIGH)
