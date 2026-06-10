import os
import cv2
import numpy as np
import re
import json

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR  = os.path.join(BASE_DIR, "logs", "crops")
FRAMES_DIR = os.path.join(BASE_DIR, "logs", "frames")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4o"

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
#  PERSPECTIVE CORRECTION
# ==============================================================

def correct_perspective(image_bgr):
    """Mild keystone correction for angled furniture photos."""
    h, w = image_bgr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, 0], [int(w * 0.92), h], [int(w * 0.08), h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (w, h))


# ==============================================================
#  WIREFRAME — 3-turn conversation, exactly like ChatGPT chat
# ==============================================================

def generate_wireframe_svg(image_bgr) -> str:
    """
    3-turn conversation that replicates how ChatGPT chat generates wireframes:
      Turn 1 — describe structure in natural language (think before drawing)
      Turn 2 — generate SVG from the description
      Turn 3 — compare SVG to image, self-correct
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    b64 = _encode_jpg(image_bgr)

    # ----------------------------------------------------------
    # TURN 1 — analyze and describe, do not draw yet
    # ----------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
                },
                {
                    "type": "text",
                    "text": (
                        "Look at this furniture image very carefully.\n"
                        "Before drawing anything, describe the exact structure:\n\n"
                        "1. Is the unit wider than tall, taller than wide, or roughly square?\n"
                        "   Give the approximate width:height ratio as a number (e.g. 0.6 for tall wardrobe, 6.0 for wide unit).\n\n"
                        "2. What is the primary layout?\n"
                        "   - HORIZONTAL_ZONES: unit has stacked rows (top zone, middle zone, bottom zone)\n"
                        "   - VERTICAL_SECTIONS: unit has side-by-side columns\n"
                        "   - BOTH: has horizontal rows AND vertical columns within rows\n\n"
                        "3. List every zone or section from top to bottom (tall units) or left to right (wide units).\n"
                        "   For each: describe exactly what is inside.\n\n"
                        "4. For each zone/section state precisely:\n"
                        "   - Type: doors / open bay / shelves / drawers / mixed\n"
                        "   - How many doors, shelves, drawers exactly\n"
                        "   - Are there vertical dividers? Where?\n"
                        "   - Are there horizontal shelves? Where (top/middle/bottom of zone)?\n"
                        "   - Is there a visible handle on each door or drawer? yes or no\n\n"
                        "5. Is there a plinth (base strip) at the bottom?\n\n"
                        "Be precise. Do not draw yet. Do not generate SVG yet."
                    )
                }
            ]
        }
    ]

    resp1 = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1000,
        messages=messages
    )
    description = resp1.choices[0].message.content.strip()
    print(f"  [turn1] description ({len(description)} chars):\n{description[:400]}\n")
    messages.append({"role": "assistant", "content": description})

    # ----------------------------------------------------------
    # TURN 2 — generate SVG from the description
    # ----------------------------------------------------------
    messages.append({
        "role": "user",
        "content": (
            "Now generate a precise flat 2D front-view wireframe SVG based on your description above.\n\n"
            "STRICT SVG RULES:\n"
            "- viewBox='0 0 960 600' width='960' height='600'\n"
            "- White background: <rect width='960' height='600' fill='white'/>\n"
            "- All shapes: fill='none'\n"
            "- Outer furniture box: stroke='#111' stroke-width='2.5'\n"
            "- Zone/section structural dividers: stroke='#111' stroke-width='2.0'\n"
            "- Internal lines (shelves, door insets, drawer dividers): stroke='#333' stroke-width='1.2'\n"
            "- 50px margin on all sides between viewBox edge and furniture box\n\n"
            "PROPORTION RULES:\n"
            "- Reproduce the exact width:height ratio you described\n"
            "- A tall wardrobe (ratio 0.6) must look tall and narrow in the SVG\n"
            "- A wide sideboard (ratio 6.0) must look wide and short in the SVG\n"
            "- Each zone height or section width must match its real proportion\n\n"
            "DRAWING RULES:\n"
            "HORIZONTAL ZONES (tall/square unit with stacked rows):\n"
            "- Draw one full-width horizontal line between each zone\n"
            "- Within each zone draw vertical dividers if present\n"
            "- Fill each zone with its content\n\n"
            "VERTICAL SECTIONS (wide unit with columns):\n"
            "- Draw one full-height vertical line between each section\n"
            "- Fill each section with its content\n\n"
            "CONTENT RULES:\n"
            "- CLOSED DOOR: one inset rect (8px padding from zone/section edges)\n"
            "  Handle: draw a short vertical bar (30px tall) ONLY if a physical handle is visible in the image\n"
            "- OPEN BAY: completely empty — no lines inside unless a shelf is present\n"
            "- SHELF: one horizontal line at the correct vertical position, spanning full zone/column width\n"
            "  Place it where it actually appears — bottom of bay = draw it low, not centered\n"
            "- DRAWERS: one inset rect per drawer + horizontal divider between drawers\n"
            "  Handle: draw a short horizontal bar (25px wide) ONLY if a physical handle is visible\n"
            "- PLINTH: one thin rect (height ~12px) below main box if visible\n\n"
            "CRITICAL DON'TS:\n"
            "- Never draw a handle that is not physically visible in the image\n"
            "- Never add decorative double borders or extra inset rectangles\n"
            "- Never draw more shelf lines than are actually visible\n"
            "- Never split a zone with a divider if no physical divider exists\n"
            "- Never make all zones equal height unless they actually are\n\n"
            "Return ONLY raw SVG. Start with <svg and end with </svg>. No markdown, no explanation."
        )
    })

    resp2 = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=4000,
        messages=messages
    )
    svg_v1 = resp2.choices[0].message.content.strip()
    print(f"  [turn2] SVG v1 ({len(svg_v1)} chars)")
    messages.append({"role": "assistant", "content": svg_v1})

    # Rasterize turn 2 SVG so turn 3 can see the rendered visual output
    import base64
    import cairosvg
    rendered_b64 = None
    _svg_for_render = svg_v1
    _m_pre = re.search(r'(<svg[\s\S]*?</svg>)', svg_v1, re.IGNORECASE)
    if _m_pre:
        _svg_for_render = _m_pre.group(1)
    try:
        _png = cairosvg.svg2png(bytestring=_svg_for_render.encode())
        rendered_b64 = base64.b64encode(_png).decode("utf-8")
        print(f"  [turn3] SVG rasterized ({len(_png)} bytes)")
    except Exception as _e:
        print(f"  [turn3] SVG rasterization failed: {_e}")

    # ----------------------------------------------------------
    # TURN 3 — compare rendered SVG to original photo, self-correct
    # ----------------------------------------------------------
    if rendered_b64 is not None:
        turn3_content = [
            {
                "type": "text",
                "text": "Image 1 is the original furniture photo and Image 2 is what your SVG wireframe looks like when rendered."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{rendered_b64}", "detail": "high"}
            },
            {
                "type": "text",
                "text": (
                    "Compare the two images carefully. Check every detail. "
                    "First does the wireframe width to height ratio match the real furniture. "
                    "Second are all sections and zones present with correct proportions. "
                    "Third are shelf lines at the correct vertical position. "
                    "Fourth are there handles drawn that are not visible in the original photo. "
                    "Fifth are any structural elements missing from the wireframe that exist in the photo. "
                    "Sixth are any extra elements drawn in the wireframe that do not exist in the photo. "
                    "Output a corrected SVG that fixes every mistake you found. "
                    "Return only raw SVG starting with svg tag and ending with closing svg tag. "
                    "No markdown, no explanation, no code fences."
                )
            }
        ]
    else:
        turn3_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
            },
            {
                "type": "text",
                "text": (
                    "Compare your SVG wireframe to the original furniture image above.\n\n"
                    "Check each of these:\n"
                    "1. PROPORTIONS — does the SVG box width:height ratio match the real furniture?\n"
                    "2. ZONE/SECTION HEIGHTS OR WIDTHS — do they match the real proportions?\n"
                    "3. SHELF POSITIONS — are shelf lines at the correct vertical position?\n"
                    "4. HANDLES — are there any handles drawn that are NOT visible in the image?\n"
                    "5. MISSING ELEMENTS — is anything from the image missing in the SVG?\n"
                    "6. EXTRA ELEMENTS — is anything in the SVG that does not exist in the image?\n\n"
                    "If you find any mistakes, output a corrected SVG.\n"
                    "If everything is correct, output the same SVG unchanged.\n\n"
                    "Return ONLY raw SVG. Start with <svg and end with </svg>. No markdown, no explanation."
                )
            }
        ]
    messages.append({"role": "user", "content": turn3_content})

    resp3 = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=4000,
        messages=messages
    )
    svg_final = resp3.choices[0].message.content.strip()
    print(f"  [turn3] SVG final ({len(svg_final)} chars)")

    # Strip any accidental markdown fences
    svg_final = re.sub(r'^```(?:svg|xml|html)?\s*', '', svg_final, flags=re.MULTILINE)
    svg_final = re.sub(r'\s*```$',                  '', svg_final, flags=re.MULTILINE).strip()

    # Extract <svg> block even if there is surrounding text
    m = re.search(r'(<svg[\s\S]*?</svg>)', svg_final, re.IGNORECASE)
    if m:
        return m.group(1)

    # Fallback: return turn 2 result if turn 3 failed
    svg_v1 = re.sub(r'^```(?:svg|xml|html)?\s*', '', svg_v1, flags=re.MULTILINE)
    svg_v1 = re.sub(r'\s*```$',                  '', svg_v1, flags=re.MULTILINE).strip()
    m2 = re.search(r'(<svg[\s\S]*?</svg>)', svg_v1, re.IGNORECASE)
    if m2:
        print("  [turn3] Falling back to turn 2 SVG")
        return m2.group(1)

    raise ValueError(f"No SVG found in any response. Last output: {svg_final[:300]}")


# ==============================================================
#  WIREFRAME CLEANUP — photo + CV wireframe → clean SVG
# ==============================================================

def clean_wireframe_svg(image_bgr: np.ndarray, wireframe_img: np.ndarray) -> str:
    """
    2-turn GPT conversation that takes:
      image_bgr    — original furniture photo (BGR)
      wireframe_img — CV edge wireframe (grayscale uint8, white bg black lines)

    GPT uses the photo as structural ground truth and the CV wireframe as a
    geometry guide, then strips noise (wood grain, texture, floor lines) and
    returns a clean geometric SVG.
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    photo_b64 = _encode_jpg(image_bgr)
    wf_b64    = _encode_jpg(cv2.cvtColor(wireframe_img, cv2.COLOR_GRAY2BGR))

    # Adaptive canvas: furniture always fills the SVG, no empty space in the middle
    MARGIN = 50
    h_wf, w_wf = wireframe_img.shape[:2]
    ar = (w_wf / h_wf) if h_wf > 0 else 1.0
    if ar >= 1.0:               # wider than tall — fix width at 960
        canvas_w  = 960
        furn_w    = canvas_w - 2 * MARGIN
        furn_h    = max(60, round(furn_w / ar))
        canvas_h  = furn_h + 2 * MARGIN
    else:                       # taller than wide — fix height at 600
        canvas_h  = 600
        furn_h    = canvas_h - 2 * MARGIN
        furn_w    = max(60, round(furn_h * ar))
        canvas_w  = furn_w + 2 * MARGIN
    viewbox   = f"0 0 {canvas_w} {canvas_h}"
    furn_desc = f"x={MARGIN} y={MARGIN} width={furn_w} height={furn_h}"

    # ----------------------------------------------------------
    # TURN 1 — analyze structure using both images, do not draw
    # ----------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I have two images:\n"
                        "IMAGE 1: Original furniture photo\n"
                        "IMAGE 2: A computer-vision edge-detected wireframe of the same furniture\n\n"
                        "The wireframe (IMAGE 2) contains noise — lines from wood grain, texture, "
                        "reflections, and the floor — that should NOT appear in a structural wireframe.\n\n"
                        "Using IMAGE 1 as ground truth, identify only the STRUCTURAL elements:\n"
                        "1. Is the unit wider than tall, taller than wide, or roughly square?\n"
                        "   Give the approximate width:height ratio (e.g. 0.6 for a tall wardrobe, "
                        "3.0 for a wide sideboard).\n"
                        "2. What structural dividers exist? (horizontal zones, vertical sections)\n"
                        "3. For each zone/section: type (doors / shelves / drawers / open bay) "
                        "and exact count of each element.\n"
                        "4. Are there any handles physically visible? Where exactly?\n"
                        "5. Is there a plinth (base strip) at the bottom?\n\n"
                        "Ignore every line in IMAGE 2 that appears to come from wood grain, "
                        "fabric texture, shadow, or the floor.\n"
                        "Do not generate SVG yet."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}", "detail": "high"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{wf_b64}", "detail": "high"}
                },
            ]
        }
    ]

    resp1 = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=900, messages=messages
    )
    description = resp1.choices[0].message.content.strip()
    print(f"  [clean_wf/t1] description ({len(description)} chars):\n{description[:300]}\n")
    messages.append({"role": "assistant", "content": description})

    # ----------------------------------------------------------
    # TURN 2 — generate clean SVG using the pre-computed adaptive canvas
    # ----------------------------------------------------------
    messages.append({
        "role": "user",
        "content": (
            "Now generate a precise flat 2D front-view wireframe SVG based on your structural analysis.\n\n"
            "STRICT SVG RULES:\n"
            f"- viewBox='{viewbox}' width='{canvas_w}' height='{canvas_h}'\n"
            f"- White background: <rect width='{canvas_w}' height='{canvas_h}' fill='white'/>\n"
            "- All shapes: fill='none'\n"
            "- Outer furniture box: stroke='#111' stroke-width='2.5'\n"
            "- Zone/section structural dividers: stroke='#111' stroke-width='2.0'\n"
            "- Internal lines (shelves, door insets, drawer dividers): stroke='#333' stroke-width='1.2'\n"
            f"- The outer furniture box MUST be exactly: {furn_desc}\n"
            "  Do not change these coordinates — the canvas was sized to match the real proportions.\n\n"
            "CONTENT RULES:\n"
            "- CLOSED DOOR: one inset rect (8px padding from zone/section edges)\n"
            "  Handle: short vertical bar (30px tall) ONLY if a physical handle is visible\n"
            "- OPEN BAY: completely empty — no lines unless a shelf is present\n"
            "- SHELF: one horizontal line at the correct vertical position\n"
            "- DRAWERS: one inset rect per drawer + horizontal divider between drawers\n"
            "  Handle: short horizontal bar (25px wide) ONLY if a physical handle is visible\n"
            "- PLINTH: thin rect (~12px) below main box if visible\n\n"
            "CRITICAL DON'TS:\n"
            "- Never draw handles that are not physically visible in the photo\n"
            "- Never add decorative double borders or extra inset rectangles\n"
            "- Never draw more shelf lines than are actually visible\n"
            "- Never split a zone with a divider if no physical divider exists\n\n"
            "Return ONLY raw SVG. Start with <svg and end with </svg>. No markdown, no explanation."
        )
    })

    resp2 = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=4000, messages=messages
    )
    svg_out = resp2.choices[0].message.content.strip()
    print(f"  [clean_wf/t2] SVG ({len(svg_out)} chars)")

    svg_out = re.sub(r'^```(?:svg|xml|html)?\s*', '', svg_out, flags=re.MULTILINE)
    svg_out = re.sub(r'\s*```$',                  '', svg_out, flags=re.MULTILINE).strip()

    m = re.search(r'(<svg[\s\S]*?</svg>)', svg_out, re.IGNORECASE)
    if m:
        return m.group(1)

    raise ValueError(f"No SVG found in response. Got: {svg_out[:300]}")


# ==============================================================
#  WIREFRAME PAIR GENERATION
# ==============================================================

def generate_wireframe_pair(img_orig, img_gen, image_name):
    print("  [wireframe] Generating original wireframe...")
    svg_orig = generate_wireframe_svg(img_orig)

    print("  [wireframe] Generating generated wireframe...")
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
    import base64
    import cairosvg
    from openai import OpenAI

    def svg_to_b64(path):
        png = cairosvg.svg2png(url=path)
        return base64.b64encode(png).decode("utf-8")

    try:
        b64_orig = svg_to_b64(orig_svg_path)
        b64_gen  = svg_to_b64(gen_svg_path)
    except Exception as e:
        print(f"  [compare] Rasterization failed: {e}")
        return {"verdict": "ERROR", "match": None, "result_path": None}

    client = OpenAI(api_key=OPENAI_API_KEY)
    print("  [compare] Asking OpenAI to compare wireframes...")
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "You are a furniture QA inspector.\n"
                "FIRST image = ORIGINAL wireframe. SECOND = GENERATED.\n"
                "Compare structure only: zone/section count, door/drawer/shelf counts per zone, proportions.\n"
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
            "raw": raw,
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

    img_orig = correct_perspective(img_orig)
    img_gen  = correct_perspective(img_gen)

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