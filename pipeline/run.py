"""
Pipeline orchestrator
----------------------
Stages:
  S1  Localize   — rembg crop to furniture
  S4  Wireframe  — Canny edges → clean + smooth wireframe
  S6  SVG        — 3-turn GPT conversation → clean SVG wireframe
  S7  Compare    — SSIM on soft edge maps → PASS / FAIL + diff image
  S8  Debug      — numbered output images

Usage:
  cd /home/amarius/Python/ImageComp
  source venv/bin/activate
  python -m pipeline.run
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from .stage1_localize  import FurnitureLocalizer
from .stage4_wireframe import WireframeGenerator
from .stage6_svg       import SvgGenerator
from .stage7_compare   import EdgeComparator
from .stage8_debug     import DebugVisualizer
from .models           import WireframeResult

# ── paths ─────────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).resolve().parent.parent
_LOGS       = _BASE / "logs"
_FRAMES     = _LOGS / "frames"
_DEBUG_ROOT = _LOGS / "debug"

# ── image pairs for batch QA ──────────────────────────────────────────────────
IMAGE_PAIRS = [
   #{"name": "cabinet_01", "original": "images/originals/5.png",  "generated": "images/generated/5.png"},
   #{"name": "cabinet_02", "original": "images/originals/1.jpg",  "generated": "images/generated/1.jpg"},
   {"name": "cabinet_03", "original": "images/originals/2.jpg",  "generated": "images/generated/2.jpg"},
    {"name": "cabinet_04", "original": "images/originals/3.jpg",  "generated": "images/generated/3.jpg"},
   #{"name": "cabinet_05", "original": "images/originals/4.jpg",  "generated": "images/generated/4.jpg"},
   #{"name": "cabinet_06", "original": "images/originals/6.jpg",  "generated": "images/generated/6.jpg"},
   #{"name": "cabinet_07", "original": "images/originals/7.jpg",  "generated": "images/generated/7.jpg"},
    {"name": "cabinet_08", "original": "images/originals/8.jpg",  "generated": "images/generated/8.jpg"},
    {"name": "cabinet_09", "original": "images/originals/9.jpg",  "generated": "images/generated/9.jpg"},
   #{"name": "cabinet_10", "original": "images/originals/10.png", "generated": "images/generated/10.png"},
   #{"name": "cabinet_11", "original": "images/originals/11.png", "generated": "images/generated/11.png"},
]


class Pipeline:
    def __init__(self, debug: bool = True):
        self.localizer  = FurnitureLocalizer(padding=20)
        self.wf_gen     = WireframeGenerator()
        self.svg_gen    = SvgGenerator()
        self.comparator = EdgeComparator()
        self.debug      = debug

    def process_image(
        self,
        image_path: str,
        name: str,
        dbg: Optional[DebugVisualizer] = None,
    ) -> WireframeResult:
        image_bgr = cv2.imread(str(_BASE / image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        # S1 — localize
        print("    [S1] localizing...")
        cropped, bbox = self.localizer.run(image_bgr)
        if dbg:
            dbg.save_01_crop(cropped)
            dbg.save_03_edges(cropped)

        # S4 — wireframe (CV edge-based)
        print("    [S4] building wireframe...")
        wf = self.wf_gen.run(cropped)

        if dbg:
            dbg.save_05_wireframe(wf.clean_img)

        # S6 — send original photo + CV wireframe to GPT for clean SVG
        print("    [S6] generating SVG wireframe...")
        wf.svg_string = self.svg_gen.run(cropped, wf.clean_img)
        _FRAMES.mkdir(parents=True, exist_ok=True)
        svg_path = _FRAMES / f"{name}.svg"
        svg_path.write_text(wf.svg_string)
        print(f"    [S6] SVG saved → {svg_path.name}")

        return wf

    def run_qa(self, original_path: str, generated_path: str, name: str) -> dict:
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  QA: {name}")
        print(f"{sep}")

        dbg_orig = DebugVisualizer(str(_DEBUG_ROOT / f"{name}_original"))  if self.debug else None
        dbg_gen  = DebugVisualizer(str(_DEBUG_ROOT / f"{name}_generated")) if self.debug else None

        print("\n  --- ORIGINAL ---")
        orig_wf = self.process_image(original_path,  f"{name}_original",  dbg_orig)

        print("\n  --- GENERATED ---")
        gen_wf  = self.process_image(generated_path, f"{name}_generated", dbg_gen)

        # S7 — compare
        print("\n    [S7] comparing wireframes...")
        _FRAMES.mkdir(parents=True, exist_ok=True)
        diff_path = str(_FRAMES / f"{name}_diff.png")
        report = self.comparator.run(orig_wf, gen_wf, diff_path=diff_path)

        # S8 — side-by-side overlay
        if self.debug and dbg_orig:
            orig_img = cv2.imread(str(_BASE / original_path))
            gen_img  = cv2.imread(str(_BASE / generated_path))
            if orig_img is not None and gen_img is not None:
                dbg_orig.save_08_overlay(orig_img, gen_img)
            _save_wireframe_overlay(orig_wf, gen_wf,
                                    str(_DEBUG_ROOT / f"{name}_original" / "08_wireframe_overlay.png"))

        icon = "PASS" if report.verdict == "PASS" else "FAIL"
        print(f"\n  Result: {icon}  SSIM={report.confidence:.1f}%")
        for d in report.differences:
            print(f"    [{d.severity.upper():8}] {d.description}")

        report_path = str(_FRAMES / f"{name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        return {
            "name":       name,
            "verdict":    report.verdict,
            "confidence": report.confidence,
            "diff":       diff_path,
            "report":     report_path,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_wireframe_overlay(orig: WireframeResult, gen: WireframeResult, path: str) -> None:
    h = max(orig.clean_img.shape[0], gen.clean_img.shape[0])

    def to_h(img):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    o_bgr = cv2.cvtColor(to_h(orig.clean_img), cv2.COLOR_GRAY2BGR)
    g_bgr = cv2.cvtColor(to_h(gen.clean_img),  cv2.COLOR_GRAY2BGR)
    gap   = np.full((h, 6, 3), 200, dtype=np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, np.hstack([o_bgr, gap, g_bgr]))


# ── convenience functions ─────────────────────────────────────────────────────

def run_qa(original_path: str, generated_path: str, name: str,
           debug: bool = True) -> dict:
    return Pipeline(debug=debug).run_qa(original_path, generated_path, name)


def run_batch(pairs=None, debug: bool = True):
    if pairs is None:
        pairs = IMAGE_PAIRS

    pipeline = Pipeline(debug=debug)
    results  = []
    for pair in pairs:
        r = pipeline.run_qa(pair["original"], pair["generated"], pair["name"])
        results.append(r)

    sep = "=" * 55
    print(f"\n{sep}")
    print("  BATCH SUMMARY")
    print(f"{sep}")
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    for r in results:
        icon = "PASS" if r["verdict"] == "PASS" else "FAIL"
        print(f"  {icon}  {r['name']:<30}  conf={r['confidence']:.1f}%")
    print(f"\n  {passed}/{len(results)} passed")
    return results


if __name__ == "__main__":
    run_batch()
