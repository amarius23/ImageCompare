"""
Stage 7 — Edge-Map Comparator
---------------------------------
Compares two WireframeResults by measuring structural similarity on their
soft maps (Gaussian-blurred edge density fields).

Steps:
  1. Resize both soft maps to the same canonical size
     (400 px wide, height = 400 / original aspect ratio).
  2. Compute SSIM — the industry-standard perceptual similarity metric.
  3. Compute a colour diff image to show WHERE the structures disagree:
       Red   = structure present in original but not in generated
       Blue  = structure present in generated but not in original
       White = both agree (no structure or matching structure)
  4. PASS  if SSIM ≥ 0.65
     FAIL  if SSIM < 0.65

The soft maps already incorporate a 14 px blur so small positional
differences (perspective, slight scale mismatch) are absorbed before
the SSIM is computed.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .models import ComparisonDifference, ComparisonReport, WireframeResult

_CANONICAL_W = 400
_PASS_SSIM   = 0.65


class EdgeComparator:
    def run(
        self,
        orig: WireframeResult,
        gen:  WireframeResult,
        diff_path: Optional[str] = None,
    ) -> ComparisonReport:
        # ── normalise to same canonical size ──────────────────────────────────
        canon_h = max(1, int(_CANONICAL_W / orig.aspect_ratio))
        o_map = cv2.resize(orig.soft_map, (_CANONICAL_W, canon_h),
                           interpolation=cv2.INTER_LINEAR)
        g_map = cv2.resize(gen.soft_map,  (_CANONICAL_W, canon_h),
                           interpolation=cv2.INTER_LINEAR)

        # ── SSIM ──────────────────────────────────────────────────────────────
        ssim_score, ssim_map = _ssim(o_map, g_map)

        # ── colour diff image ─────────────────────────────────────────────────
        diff_bgr = _diff_image(o_map, g_map)
        if diff_path:
            Path(diff_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(diff_path, diff_bgr)

        # ── aspect ratio difference ───────────────────────────────────────────
        diffs = []
        ratio_err = abs(orig.aspect_ratio - gen.aspect_ratio) / max(orig.aspect_ratio, 0.1)
        if ratio_err > 0.18:
            diffs.append(ComparisonDifference(
                type="aspect_ratio",
                description=(f"Aspect ratio: orig={orig.aspect_ratio:.2f} "
                             f"gen={gen.aspect_ratio:.2f}  "
                             f"({ratio_err*100:.0f}% difference)"),
                severity="warning",
                original=round(orig.aspect_ratio, 2),
                generated=round(gen.aspect_ratio, 2),
            ))

        if ssim_score < _PASS_SSIM:
            diffs.append(ComparisonDifference(
                type="structural_similarity",
                description=(f"SSIM={ssim_score:.3f} is below the pass threshold "
                             f"of {_PASS_SSIM:.2f}"),
                severity="critical",
                original=_PASS_SSIM,
                generated=round(ssim_score, 3),
            ))

        verdict    = "PASS" if ssim_score >= _PASS_SSIM else "FAIL"
        confidence = round(min(100.0, ssim_score * 100), 1)

        return ComparisonReport(
            verdict=verdict,
            confidence=confidence,
            differences=diffs,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _ssim(a: np.ndarray, b: np.ndarray, window: int = 11) -> tuple[float, np.ndarray]:
    """
    Compute SSIM between two float32 [0–1] grayscale maps.
    Returns (scalar_score, full_ssim_map).
    Falls back to simple correlation if skimage is unavailable.
    """
    try:
        from skimage.metrics import structural_similarity
        score, ssim_map = structural_similarity(
            a, b, data_range=1.0, win_size=window, full=True
        )
        return float(score), ssim_map.astype(np.float32)
    except ImportError:
        # Fallback: normalised cross-correlation
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        numer = np.sum((a_f - a_f.mean()) * (b_f - b_f.mean()))
        denom = np.sqrt(np.sum((a_f - a_f.mean())**2) * np.sum((b_f - b_f.mean())**2))
        score = float(numer / denom) if denom > 0 else 0.0
        return score, np.ones_like(a)


def _diff_image(orig: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """
    Produce a BGR colour diff image:
      Red   = structure only in original
      Blue  = structure only in generated
      White = background / matching
    """
    h, w = orig.shape
    out = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    only_orig = np.clip(orig - gen, 0, None)
    only_gen  = np.clip(gen - orig, 0, None)

    # Normalise to [0, 255] using the combined maximum so colours are comparable
    mx = max(only_orig.max(), only_gen.max(), 1e-6)
    orig_u8 = (only_orig / mx * 255).astype(np.uint8)
    gen_u8  = (only_gen  / mx * 255).astype(np.uint8)

    # Red channel → original structure missing in generated
    out[:, :, 2] = np.maximum(out[:, :, 2].astype(int) - orig_u8, 0).astype(np.uint8)
    out[:, :, 1] = np.maximum(out[:, :, 1].astype(int) - orig_u8, 0).astype(np.uint8)

    # Blue channel → generated structure missing in original
    out[:, :, 0] = np.maximum(out[:, :, 0].astype(int) - gen_u8, 0).astype(np.uint8)
    out[:, :, 1] = np.maximum(out[:, :, 1].astype(int) - gen_u8, 0).astype(np.uint8)

    return out
