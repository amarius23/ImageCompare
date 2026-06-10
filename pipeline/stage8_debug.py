import cv2
import numpy as np
from pathlib import Path


class DebugVisualizer:
    def __init__(self, debug_dir: str):
        self.out = Path(debug_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def save_01_crop(self, image: np.ndarray) -> str:
        return self._save("01_crop.png", image)

    def save_03_edges(self, image: np.ndarray) -> str:
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 100)
        return self._save("03_edges.png", edges)

    def save_05_wireframe(self, wireframe: np.ndarray) -> str:
        return self._save("05_wireframe.png", wireframe)

    def save_08_overlay(self, orig: np.ndarray, gen: np.ndarray) -> str:
        h = max(orig.shape[0], gen.shape[0])

        def resize_to_height(img, target_h):
            scale = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), target_h))

        orig_r = resize_to_height(orig, h)
        gen_r  = resize_to_height(gen, h)
        gap    = np.full((h, 4, 3), 200, dtype=np.uint8)
        return self._save("08_overlay.png", np.hstack([orig_r, gap, gen_r]))

    def _save(self, name: str, image: np.ndarray) -> str:
        path = str(self.out / name)
        cv2.imwrite(path, image)
        return path
