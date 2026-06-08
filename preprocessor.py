import os
import cv2
import numpy as np
import re
import json

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR  = os.path.join(BASE_DIR, "logs", "crops")
FRAMES_DIR = os.path.join(BASE_DIR, "logs", "frames")

OPENAI_API_KEY = ""
OPENAI_MODEL   = "gpt-5.4"

# ==============================================================
#  WIREFRAME PROMPT — OpenAI generates SVG directly
# ==============================================================

WIREFRAME_PROMPT = (
    "You are a precise technical wireframe generator for furniture QA.\n"
    "Study this furniture image and generate a flat 2D front-view wireframe SVG.\n\n"

    "STRICT SVG RULES:\n"
    "- viewBox='0 0 960 600' width='960' height='600'\n"
    "- White background: <rect width='960' height='600' fill='white'/>\n"
    "- All shapes: fill='none'\n"
    "- Outer box: stroke='#111' stroke-width='2.5'\n"
    "- Main section dividers (full height): stroke='#111' stroke-width='2.0'\n"
    "- Internal shelf lines and sub-dividers: stroke='#333' stroke-width='1.2'\n"
    "- Door/drawer inset rects: stroke='#333' stroke-width='1.2'\n"
    "- 50px margin from viewBox edges to outer furniture box\n"
    "- font-family='Arial,sans-serif' font-size='11' fill='#444' for labels below box\n\n"

    "STEP 1 — SECTION ISOLATION RULE:\n"
    "This is the most important rule:\n"
    "- Every vertical section is a completely independent zone\n"
    "- Internal dividers, shelves, or details inside one section must NOT continue into any adjacent section\n"
    "- Each section's internal lines start and end exactly at that section's left and right boundaries\n"
    "- Count sections from left to right, assign each a width_pct, and draw them independently\n\n"

    "STEP 2 — MEASURE PRECISELY:\n"
    "- Measure unit width:height ratio — for a very wide low unit this could be 7:1 or more\n"
    "- Measure each section width as % of total width — write them down mentally before drawing\n"
    "- All section width_pct values must sum to exactly 100\n"
    "- Reproduce these proportions faithfully in the SVG\n\n"

    "STEP 3 — CLASSIFY EACH SECTION independently left to right:\n"
    "For each section state:\n"
    "  - type: closed_doors / open_shelf / open_bay / panel_door / drawers / mixed\n"
    "  - width_pct\n"
    "  - internal details: how many doors, shelves, dividers, drawers\n"
    "  - handles: visible yes/no\n\n"

    "STEP 4 — DRAW RULES per section type:\n\n"

    "CLOSED DOORS:\n"
    "- Full height section\n"
    "- N doors: draw N-1 vertical dividers evenly spaced within section\n"
    "- One inset rect per door (8px padding from section boundaries)\n"
    "- Handle only if physically visible in image\n\n"

    "OPEN SHELF WITH TOP BAY + VERTICAL DIVIDERS:\n"
    "- Draw ONE horizontal shelf line at the correct height separating top open bay from lower divider zone\n"
    "- Draw vertical dividers ONLY below that shelf line, within this section only\n"
    "- Count dividers exactly — count the lines not the slots\n"
    "- Dividers go from the shelf line to the bottom of the section\n"
    "- The top open bay has NO vertical dividers\n"
    "- Any additional shelves within sub-bays: draw as horizontal lines contained within their sub-bay only\n\n"

    "OPEN BAY (no dividers):\n"
    "- Empty interior\n"
    "- If one shelf visible: draw one horizontal line at correct vertical position\n"
    "- If a partial vertical divider visible: draw it contained within this section only\n\n"

    "PANEL DOOR (flat flush door, no frame):\n"
    "- Draw ONE inset rect (8px padding) for the full section height\n"
    "- No additional internal lines\n"
    "- No handle unless physically visible\n\n"

    "DRAWERS (stacked):\n"
    "- Count drawers exactly\n"
    "- Divide section height equally per drawer\n"
    "- Draw horizontal divider lines between drawers\n"
    "- One inset rect per drawer (6px padding)\n"
    "- Handle only if physically visible\n\n"

    "MIXED (panel door upper + drawers lower):\n"
    "- Draw ONE horizontal line separating panel zone from drawer zone at correct height\n"
    "- Upper: panel door inset rect\n"
    "- Lower: drawer dividers + inset rects\n"
    "- No handles unless visible\n\n"

    "PLINTH:\n"
    "- One thin rect below main box if visible (height ~10px)\n\n"

    "ABSOLUTE DON'T RULES:\n"
    "- NEVER let internal lines from one section cross into another section\n"
    "- NEVER continue vertical dividers past a section boundary\n"
    "- NEVER draw more shelf lines than are visible\n"
    "- NEVER draw handles that are not visible\n"
    "- NEVER add decorative borders or double insets\n"
    "- NEVER make sections equal width unless they actually are\n"
    "- NEVER draw horizontal rows across the full unit unless a physical full-width panel exists\n\n"

    "OUTPUT:\n"
    "- Return ONLY raw SVG — no markdown, no explanation, no code fences\n"
    "- Start with <svg and end with </svg>"
)

# ==============================================================
#  UTILITIES
# ==============================================================

def _ensure_dirs():
    os.makedirs(CROPS_DIR,  exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def _encode_jpg(image_bgr):
    import base64
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ==============================================================
#  LOCALIZATION — rembg background removal → tight crop
# ==============================================================

def crop_to_furniture(image_path, padding=20):
    img = load_image(image_path)
    h_img, w_img = img.shape[:2]
    try:
        from rembg import remove as rembg_remove
        from PIL import Image as PILImage
        alpha = np.array(rembg_remove(
            PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ))[:, :, 3]
        mask = np.where(alpha > 127, 255, 0).astype(np.uint8)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels < 2:
            return img
        cx_img, cy_img = w_img / 2.0, h_img / 2.0
        diag = (w_img ** 2 + h_img ** 2) ** 0.5
        best_lbl, best_score = 1, -1.0
        for lbl in range(1, n_labels):
            s  = stats[lbl]
            cx = s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]  / 2.0
            cy = s[cv2.CC_STAT_TOP]  + s[cv2.CC_STAT_HEIGHT] / 2.0
            dist  = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
            score = float(s[cv2.CC_STAT_AREA]) * (1.0 - dist / diag) ** 4
            if score > best_score:
                best_score, best_lbl = score, lbl
        s = stats[best_lbl]
        x = max(0, s[cv2.CC_STAT_LEFT]   - padding)
        y = max(0, s[cv2.CC_STAT_TOP]    - padding)
        w = min(w_img - x, s[cv2.CC_STAT_WIDTH]  + padding * 2)
        h = min(h_img - y, s[cv2.CC_STAT_HEIGHT] + padding * 2)
        print(f"  [rembg] bbox x={x} y={y} w={w} h={h}")
        return img[y:y+h, x:x+w]
    except Exception as e:
        print(f"  [rembg] Failed ({e}) — using full image.")
        return img


# ==============================================================
#  WIREFRAME — OpenAI generates SVG directly from image
# ==============================================================

def generate_wireframe_svg(image_bgr):
    """Ask OpenAI to generate a wireframe SVG directly from the image."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    b64 = _encode_jpg(image_bgr)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_completion_tokens=4000,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": WIREFRAME_PROMPT},
            ]}],
        )
        choice = resp.choices[0]
        print(f"  [wireframe] finish_reason: {choice.finish_reason}")
        raw = (choice.message.content or "").strip()
        print(f"  [wireframe] response ({len(raw)} chars): {raw[:200]}")
    except Exception as e:
        print(f"  [wireframe] API call failed: {e}")
        raise

    # Strip any accidental markdown fences
    raw = re.sub(r'^```(?:svg|xml)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$',             '', raw, flags=re.MULTILINE).strip()

    if not raw.startswith("<svg"):
        raise ValueError(f"Response is not SVG: {raw[:200]}")

    return raw


def generate_wireframe_pair(img_orig, img_gen, image_name):
    """Ask OpenAI to generate SVG wireframes directly for both images."""
    print(f"  [wireframe] Generating original wireframe...")
    svg_orig = generate_wireframe_svg(img_orig)

    print(f"  [wireframe] Generating generated wireframe...")
    svg_gen = generate_wireframe_svg(img_gen)

    orig_path = os.path.join(FRAMES_DIR, f"{image_name}_original_frame.svg")
    gen_path  = os.path.join(FRAMES_DIR, f"{image_name}_generated_frame.svg")

    with open(orig_path, "w") as f: f.write(svg_orig)
    with open(gen_path,  "w") as f: f.write(svg_gen)

    print(f"  [wireframe] Saved → {orig_path}")
    print(f"  [wireframe] Saved → {gen_path}")
    return orig_path, gen_path


# ==============================================================
#  COMPARISON — OpenAI compares both rendered SVG images
# ==============================================================

def compare_wireframes(orig_svg_path, gen_svg_path, image_name):
    import base64, cairosvg
    from openai import OpenAI

    def svg_to_b64(path):
        png = cairosvg.svg2png(url=path, output_width=960, output_height=600)
        return base64.b64encode(png).decode("utf-8")

    try:
        b64_orig = svg_to_b64(orig_svg_path)
        b64_gen  = svg_to_b64(gen_svg_path)
    except Exception as e:
        print(f"  [compare] Rasterization failed: {e}")
        return {"verdict": "ERROR", "match": None, "result_path": None}

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"  [compare] Asking OpenAI to compare wireframes...")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=1200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "You are a furniture QA inspector. FIRST image = ORIGINAL wireframe. SECOND = GENERATED.\n"
                "Compare structure only: section count, door/drawer/shelf counts per section, proportions.\n"
                'Reply ONLY in JSON: {"verdict":"MATCH" or "NO MATCH","confidence":0-100,'
                '"differences":["..."],"summary":"one sentence"}'
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_orig}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_gen}",  "detail": "high"}},
        ]}],
    )

    raw = resp.choices[0].message.content.strip()
    print(f"  [compare] Raw: {raw}")
    m = re.search(r'\{[\s\S]*\}', raw)
    try:
        result = json.loads(m.group(0) if m else raw)
    except Exception:
        upper  = raw.upper()
        result = {
            "verdict": "NO MATCH" if "NO MATCH" in upper else "MATCH" if "MATCH" in upper else "UNKNOWN",
            "raw": raw
        }

    verdict = result.get("verdict", "UNKNOWN")
    print(f"  [compare] {verdict}  confidence={result.get('confidence','?')}%  — {result.get('summary','')}")

    out_path = os.path.join(FRAMES_DIR, f"{image_name}_comparison_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return {"verdict": verdict, "match": verdict == "MATCH", "result_path": out_path}


# ==============================================================
#  PIPELINE
# ==============================================================

def run_qa(original_path, generated_path, auto_bbox=True):
    image_name = os.path.splitext(os.path.basename(original_path))[0]
    print("\n" + "=" * 55)
    print(f"  Processing: {image_name}")
    print("=" * 55)

    _ensure_dirs()

    img_orig = crop_to_furniture(original_path)  if auto_bbox else load_image(original_path)
    img_gen  = crop_to_furniture(generated_path) if auto_bbox else load_image(generated_path)

    orig_crop = os.path.join(CROPS_DIR, f"{image_name}_original.png")
    gen_crop  = os.path.join(CROPS_DIR, f"{image_name}_generated.png")
    cv2.imwrite(orig_crop, img_orig)
    cv2.imwrite(gen_crop,  img_gen)
    print(f"  Crops → {orig_crop}, {gen_crop}")

    orig_svg, gen_svg = generate_wireframe_pair(img_orig, img_gen, image_name)

    cmp = compare_wireframes(orig_svg, gen_svg, image_name) if orig_svg and gen_svg else \
          {"verdict": "ERROR", "match": None, "result_path": None}

    return {
        "image_name" : image_name,
        "orig_crop"  : orig_crop,
        "gen_crop"   : gen_crop,
        "orig_svg"   : orig_svg,
        "gen_svg"    : gen_svg,
        "verdict"    : cmp["verdict"],
        "match"      : cmp["match"],
        "result_path": cmp["result_path"],
    }


def run_batch(pairs):
    results = []
    for pair in pairs:
        r = run_qa(pair["original"], pair["generated"], auto_bbox=pair.get("auto_bbox", True))
        r["image_name"] = pair["name"]
        results.append(r)

    print("\n" + "=" * 55)
    for r in results:
        icon = "✅" if r["match"] else ("❌" if r["match"] is False else "⚠️")
        print(f"  {icon}  {r['image_name']:30}  {r['verdict']}")
        print(f"       {r['result_path']}")
    print("=" * 55)
    return results


# ==============================================================
#  IMAGE PAIRS
# ==============================================================

IMAGE_PAIRS = [
    # {"name": "cabinet_01", "original": "images/originals/5.png",  "generated": "images/generated/5.png",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_02", "original": "images/originals/1.jpg",  "generated": "images/generated/1.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_03", "original": "images/originals/2.jpg",  "generated": "images/generated/2.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_04", "original": "images/originals/3.jpg",  "generated": "images/generated/3.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_05", "original": "images/originals/4.jpg",  "generated": "images/generated/4.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_06", "original": "images/originals/6.jpg",  "generated": "images/generated/6.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_07", "original": "images/originals/7.jpg",  "generated": "images/generated/7.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_08", "original": "images/originals/8.jpg",  "generated": "images/generated/8.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_09", "original": "images/originals/9.jpg",  "generated": "images/generated/9.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_10", "original": "images/originals/10.png", "generated": "images/generated/10.png", "bbox": None, "auto_bbox": True, "debug_bbox": True},
    # {"name": "cabinet_11", "original": "images/originals/11.png", "generated": "images/generated/11.png", "bbox": None, "auto_bbox": True, "debug_bbox": True},
]

if __name__ == "__main__":
    run_batch(IMAGE_PAIRS)