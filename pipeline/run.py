"""
Pipeline orchestrator
---------------------
Stages:
  S1   Crop         — smart floor/wall trim (rembg or edge-density fallback)
  S1b  Perspective  — detect frontal vs angled view; auto NO MATCH if mismatched
  S2   Vision       — GPT-4.1 vision → structured JSON
  S3   Describe     — JSON → concise text description
  S4   Edges        — OpenCV Canny + Hough lines → edge map
  S5   Wireframe    — gpt-image-1 image editing → wireframe PNG
  S6   Compare      — GPT-4.1 vision: original vs generated wireframe verdict

Each cabinet is processed as a pair (original + generated).

Output: logs/wireframes/{name}/original/  and  logs/wireframes/{name}/generated/
  1_cropped.png
  1b_perspective.json
  2_vision.json
  3_description.txt
  4_edges.png
  5_wireframe.png
logs/wireframes/{name}/6_comparison.json

Usage:
  cd /home/amarius/Python/ImageComp
  source venv/bin/activate
  python -m pipeline.run                          # full batch
  python -m pipeline.run images/originals/2.jpg  # single image (no pair)
"""

import json
import os
import sys
import cv2
from pathlib import Path
from openai import OpenAI

from .stage1_crop         import crop_furniture
from .stage1b_perspective import check_perspective
from .stage2_vision       import analyze_vision
from .stage3_describe     import build_description
from .stage4_edges        import extract_edges
from .stage5_wireframe    import generate_wireframe
from .stage6_compare      import compare_wireframes

_BASE = Path(__file__).resolve().parent.parent
_LOGS = _BASE / "logs" / "wireframes"

# ── env setup ─────────────────────────────────────────────────────────────────

def _load_env() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_file = _BASE / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()
_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── core single-image runner ──────────────────────────────────────────────────

def _run_image(image_path: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(_BASE / image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # S1
    print("    [S1] cropping...")
    crop = crop_furniture(image_bgr)
    cv2.imwrite(str(out_dir / "1_cropped.png"), crop.image)
    print(f"         {crop.w}×{crop.h}  →  1_cropped.png")

    # S1b
    print("    [S1b] perspective check...")
    perspective = check_perspective(crop.image, _CLIENT)
    (out_dir / "1b_perspective.json").write_text(json.dumps(perspective, indent=2))

    # S2
    print("    [S2] vision analysis (GPT-4.1)...")
    vision_json = analyze_vision(crop.image, _CLIENT)
    (out_dir / "2_vision.json").write_text(json.dumps(vision_json, indent=2))
    n_sections = len(vision_json.get("sections", []))
    print(f"         {n_sections} sections  →  2_vision.json")

    # S3
    print("    [S3] building description...")
    description = build_description(vision_json)
    (out_dir / "3_description.txt").write_text(description)
    print("         3_description.txt")

    # S4
    print("    [S4] extracting edges (Canny + Hough)...")
    edge_map = extract_edges(crop.image)
    cv2.imwrite(str(out_dir / "4_edges.png"), edge_map)
    print("         4_edges.png")

    # S5
    print("    [S5] generating wireframe (gpt-image-1)...")
    wireframe = generate_wireframe(crop.image, edge_map, description, _CLIENT, vision_json)
    wireframe_path = str(out_dir / "5_wireframe.png")
    cv2.imwrite(wireframe_path, wireframe)
    print("         5_wireframe.png")

    return {"wireframe": wireframe_path, "sections": n_sections, "perspective": perspective}


# ── public API ────────────────────────────────────────────────────────────────

def run_pair(name: str, original_path: str, generated_path: str) -> dict:
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  Cabinet: {name}")
    print(f"{sep}")

    base = _LOGS / name

    print("\n  -- original --")
    orig = _run_image(original_path, base / "original")

    print("\n  -- generated --")
    gen  = _run_image(generated_path, base / "generated")

    # S6 — compare the two wireframes
    print("\n  -- comparison --")

    orig_persp = orig.get("perspective", {})
    gen_persp  = gen.get("perspective",  {})
    orig_front = orig_persp.get("is_frontal", True)
    gen_front  = gen_persp.get("is_frontal",  True)

    if not orig_front or not gen_front:
        # One or both shots are angled — wireframe comparison is meaningless
        mismatch_reason = []
        if not orig_front:
            mismatch_reason.append(
                f"original is {orig_persp.get('view_type','angled')} (not frontal)"
            )
        if not gen_front:
            mismatch_reason.append(
                f"generated is {gen_persp.get('view_type','angled')} (not frontal)"
            )
        comparison = {
            "verdict":    "NO MATCH",
            "confidence": 0,
            "reason":     "perspective mismatch — " + "; ".join(mismatch_reason),
            "differences": mismatch_reason,
            "matching_elements": [],
            "summary": (
                "Comparison skipped: the furniture is not photographed "
                "straight-on so wireframes cannot be structurally compared."
            ),
        }
        print(f"    [S6] NO MATCH — {comparison['reason']}")
    else:
        orig_wf = cv2.imread(str(base / "original"  / "5_wireframe.png"))
        gen_wf  = cv2.imread(str(base / "generated" / "5_wireframe.png"))
        if orig_wf is not None and gen_wf is not None:
            comparison = compare_wireframes(orig_wf, gen_wf, _CLIENT)
        else:
            comparison = {"error": "one or both wireframe images missing"}
            print("    [S6] skipped — wireframe files not found")

    comp_path = base / "6_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2))
    print("         6_comparison.json")

    print(f"\n  Done → {base}")
    return {"name": name, "original": orig, "generated": gen, "comparison": comparison}


def run_single(image_path: str, name: str | None = None) -> dict:
    path = _BASE / image_path
    if name is None:
        name = path.stem
    out_dir = _LOGS / name
    print(f"\n{'=' * 55}")
    print(f"  Image: {name}")
    print(f"{'=' * 55}")
    r = _run_image(image_path, out_dir)
    print(f"\n  Done → {out_dir}")
    return {"name": name, **r}


def run_batch(pairs: list[dict] | None = None) -> list[dict]:
    if pairs is None:
        pairs = IMAGE_PAIRS

    results = []
    for entry in pairs:
        r = run_pair(entry["name"], entry["original"], entry["generated"])
        results.append(r)

    sep = "=" * 55
    print(f"\n{sep}")
    print("  BATCH SUMMARY")
    print(f"{sep}")
    for r in results:
        comp     = r.get("comparison", {})
        verdict  = comp.get("verdict",    "—")
        conf     = comp.get("confidence", "—")
        ssim     = comp.get("ssim_score", "—")
        print(f"  {r['name']}  →  {verdict}  ({conf}%  ssim={ssim})")
        print(f"    original  →  {r['original']['wireframe']}")
        print(f"    generated →  {r['generated']['wireframe']}")
    print(f"\n  {len(results)} pairs processed")
    return results


# ── image pairs ───────────────────────────────────────────────────────────────

IMAGE_PAIRS = [
    {"name": "cabinet_01", "original": "images/originals/1.jpg",  "generated": "images/generated/1.jpg"},
    {"name": "cabinet_02", "original": "images/originals/2.jpg",  "generated": "images/generated/2.jpg"},
    {"name": "cabinet_03", "original": "images/originals/3.jpg",  "generated": "images/generated/3.jpg"},
    {"name": "cabinet_04", "original": "images/originals/4.jpg",  "generated": "images/generated/4.jpg"},
    {"name": "cabinet_05", "original": "images/originals/5.png",  "generated": "images/generated/5.png"},
    {"name": "cabinet_06", "original": "images/originals/6.jpg",  "generated": "images/generated/6.jpg"},
    {"name": "cabinet_07", "original": "images/originals/7.jpg",  "generated": "images/generated/7.jpg"},
    {"name": "cabinet_08", "original": "images/originals/8.jpg",  "generated": "images/generated/8.jpg"},
    {"name": "cabinet_09", "original": "images/originals/9.jpg",  "generated": "images/generated/9.jpg"},
    {"name": "cabinet_10", "original": "images/originals/10.png", "generated": "images/generated/10.png"},
    {"name": "cabinet_11", "original": "images/originals/11.png", "generated": "images/generated/11.png"},
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(sys.argv[1])
    else:
        run_batch()
