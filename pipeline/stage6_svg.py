import cv2
import numpy as np
import json
import base64
import os
from openai import OpenAI

OPENAI_MODEL = "gpt-4o"


def _encode_jpg(image_bgr):
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ==============================================================
#  STEP 1 — Python detects structural lines from the wireframe
#  This gives us exact pixel positions with zero model involvement
# ==============================================================

def detect_structural_lines(wireframe_img: np.ndarray) -> dict:
    """
    Use morphological line detection to find the main
    vertical and horizontal structural lines in the wireframe.
    Returns pixel positions of dividers.
    """
    h, w = wireframe_img.shape[:2]

    # Convert to grayscale and threshold
    if len(wireframe_img.shape) == 3:
        gray = cv2.cvtColor(wireframe_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = wireframe_img.copy()

    # Invert if background is white
    if gray.mean() > 127:
        gray = cv2.bitwise_not(gray)

    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # --- Detect vertical lines ---
    # Kernel height = 60% of image height minimum
    vk_h = max(int(h * 0.6), 40)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk_h))
    v_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Find x positions of vertical lines via column projection
    v_proj = v_lines.sum(axis=0).astype(np.float32)
    v_proj = v_proj / v_proj.max() if v_proj.max() > 0 else v_proj

    # Find peaks in projection (each peak = one vertical line)
    vertical_x = _find_peaks(v_proj, min_val=0.3, min_dist=int(w * 0.04))

    # --- Detect horizontal lines ---
    hk_w = max(int(w * 0.3), 40)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk_w, 1))
    h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    h_proj = h_lines.sum(axis=1).astype(np.float32)
    h_proj = h_proj / h_proj.max() if h_proj.max() > 0 else h_proj

    horizontal_y = _find_peaks(h_proj, min_val=0.3, min_dist=int(h * 0.04))

    # Convert to percentages
    v_pcts = sorted([round(x / w * 100, 1) for x in vertical_x])
    h_pcts = sorted([round(y / h * 100, 1) for y in horizontal_y])

    # Remove outer box edges (near 0% and 100%)
    v_pcts = [p for p in v_pcts if 2 < p < 98]
    h_pcts = [p for p in h_pcts if 2 < p < 98]

    print(f"  [lines] vertical dividers at: {v_pcts}")
    print(f"  [lines] horizontal shelves at: {h_pcts}")

    return {
        "vertical_dividers_pct":  v_pcts,
        "horizontal_shelves_pct": h_pcts,
        "image_width":  w,
        "image_height": h,
    }


def _find_peaks(projection, min_val=0.3, min_dist=20):
    """Find peak positions in a 1D projection array."""
    peaks = []
    n = len(projection)
    i = 0
    while i < n:
        if projection[i] >= min_val:
            # Find the center of this peak cluster
            j = i
            while j < n and projection[j] >= min_val:
                j += 1
            center = (i + j) // 2
            # Check minimum distance from last peak
            if not peaks or (center - peaks[-1]) >= min_dist:
                peaks.append(center)
            i = j
        else:
            i += 1
    return peaks


# ==============================================================
#  STEP 2 — Model classifies content between detected lines
#  Model gets the photo + the section boundaries
#  Model only answers WHAT is in each section, not WHERE
# ==============================================================

def classify_sections(
    photo_bgr: np.ndarray,
    line_data: dict
) -> dict:
    client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    photo_b64 = _encode_jpg(photo_bgr)
    h, w      = photo_bgr.shape[:2]

    v_pcts     = line_data["vertical_dividers_pct"]
    boundaries = [0.0] + v_pcts + [100.0]
    sections   = []

    # ── classify each section individually from its crop ──────
    for i in range(len(boundaries) - 1):
        x0_pct = boundaries[i]
        x1_pct = boundaries[i + 1]
        sec_w   = x1_pct - x0_pct

        if sec_w < 4.0:
            print(f"  [classify] skipping S{i+1} — too narrow ({sec_w:.1f}%)")
            continue

        px0      = max(0, int(x0_pct / 100 * w))
        px1      = min(w, int(x1_pct / 100 * w))
        crop     = photo_bgr[:, px0:px1]
        crop_b64 = _encode_jpg(crop)

        prompt = f"""You are a furniture section classifier.

This image shows ONE section of a furniture unit.
This section occupies {round(sec_w, 1)}% of the total furniture width.

Look at this section carefully and answer:

1. TYPE — what is the primary content?
   closed_door: has one or more closed flat door panels covering the section
   open_shelf: has a wide open area at top AND narrow vertical slot dividers below a horizontal shelf
   open_bay: fully open with no doors, no slot dividers — just empty space possibly with one shelf
   drawers: has stacked horizontal drawer fronts
   mixed: combination (open top + drawers at bottom, or similar)

2. DOORS — how many individual door panels? (0 if not closed_door)

3. DRAWERS — how many individual drawers? (0 if drawers or mixed)

4. SLOT_DIVIDERS — if type is open_shelf: how many narrow vertical divider LINES are visible below the shelf?
   Count the lines, not the spaces. If type is not open_shelf write 0.

5. SHELF_Y_PCT — if there is a horizontal shelf inside this section, at approximately what
   percentage from the TOP of the full furniture unit does it sit?
   E.g. if the shelf is about halfway down the unit write 50.
   Write null if no shelf.

6. HAS_HANDLE — is there a visible physical handle or knob on any door or drawer? true or false.

Return JSON only. No markdown.
{{"type":"...","doors":0,"drawers":0,"slot_dividers":0,"shelf_y_pct":null,"has_handle":false}}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}",
                                   "detail": "high"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        raw = resp.choices[0].message.content.strip()
        print(f"  [classify] S{i+1} ({round(sec_w,1)}%): {raw}")

        try:
            sec_data = json.loads(raw)
        except json.JSONDecodeError:
            sec_data = {"type": "open_bay", "doors": 0, "drawers": 0,
                        "slot_dividers": 0, "shelf_y_pct": None, "has_handle": False}

        sections.append({
            "section_id":    f"S{i+1}",
            "x_start_pct":   round(x0_pct, 1),
            "x_end_pct":     round(x1_pct, 1),
            "type":          sec_data.get("type", "open_bay"),
            "doors":         int(sec_data.get("doors", 0)),
            "drawers":       int(sec_data.get("drawers", 0)),
            "slot_count":    int(sec_data.get("slot_dividers", 0)),
            "shelf_y_pct":   sec_data.get("shelf_y_pct"),
            "has_handle":    bool(sec_data.get("has_handle", False)),
            "shelves":       [],
            "slot_dividers": [],
        })

    # ── ask model for overall dimensions once ─────────────────
    dim_resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=100,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}",
                               "detail": "high"}},
                {"type": "text", "text": (
                    "Look at this furniture. Measure the cabinet body only, ignore the room.\n"
                    "Return JSON only:\n"
                    '{"width_to_height_ratio": <float>, "has_plinth": <bool>}'
                )}
            ]
        }]
    )
    dim_raw = dim_resp.choices[0].message.content.strip()
    print(f"  [classify] dimensions: {dim_raw}")
    try:
        dims = json.loads(dim_raw)
    except Exception:
        dims = {"width_to_height_ratio": 4.0, "has_plinth": True}

    return {
        "width_to_height_ratio": dims.get("width_to_height_ratio", 4.0),
        "has_plinth":            dims.get("has_plinth", True),
        "sections":              sections,
    }


# ==============================================================
#  STEP 3 — Python builds slot divider positions from slot_count
#  Model said "4 slot dividers" — Python computes their x positions
# ==============================================================

def _expand_slots(structure: dict) -> dict:
    """
    Convert slot_count integers to actual slot_dividers with x_pct values.
    Python computes the evenly-spaced positions.
    """
    for sec in structure.get("sections", []):
        slot_count = int(sec.pop("slot_count", 0))
        shelf_y    = sec.pop("shelf_y_pct", None)

        sec.setdefault("shelves", [])
        sec.setdefault("slot_dividers", [])

        x0 = float(sec.get("x_start_pct", 0))
        x1 = float(sec.get("x_end_pct",  100))
        sec_w = x1 - x0

        # Add horizontal shelf if present
        if shelf_y is not None:
            sec["shelves"].append({
                "y_pct":       float(shelf_y),
                "x_start_pct": x0,
                "x_end_pct":   x1
            })

        # Compute evenly spaced slot divider positions
        if slot_count > 0:
            y_start = float(shelf_y) if shelf_y is not None else 0.0
            n_dividers = slot_count
            for i in range(1, n_dividers + 1):
                x_pct = round(x0 + sec_w * i / (n_dividers + 1), 1)
                sec["slot_dividers"].append({
                    "x_pct":       x_pct,
                    "y_start_pct": y_start,
                    "y_end_pct":   100.0
                })

    return structure


# ==============================================================
#  STEP 4 — Python builds SVG from structure
# ==============================================================

_MARGIN  = 50
_PADDING = 8
_MAX_DIM = 860


def _canvas(ratio: float):
    ratio = max(0.1, float(ratio))
    if ratio >= 1.0:
        furn_w = _MAX_DIM
        furn_h = round(furn_w / ratio)
    else:
        furn_h = _MAX_DIM
        furn_w = round(furn_h * ratio)
    furn_h = max(60, min(700, furn_h))
    furn_w = max(60, min(900, furn_w))
    canvas_w = furn_w + 2 * _MARGIN
    canvas_h = furn_h + 2 * _MARGIN
    x0 = _MARGIN
    y0 = _MARGIN
    x1 = x0 + furn_w
    y1 = y0 + furn_h
    return canvas_w, canvas_h, furn_w, furn_h, x0, y0, x1, y1


def build_svg(structure: dict) -> str:
    ratio      = float(structure.get("width_to_height_ratio", 4.0))
    has_plinth = structure.get("has_plinth", False)
    sections   = structure.get("sections", [])

    canvas_w, canvas_h, furn_w, furn_h, x0, y0, x1, y1 = _canvas(ratio)

    def xp(pct): return x0 + round(float(pct) / 100.0 * furn_w)
    def yp(pct): return y0 + round(float(pct) / 100.0 * furn_h)

    el = []

    def L(ax1, ay1, ax2, ay2, stroke="#1a1a1a", sw=2.0):
        el.append(f'<line x1="{ax1}" y1="{ay1}" x2="{ax2}" y2="{ay2}" '
                  f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"/>')

    def R(rx, ry, rw, rh, stroke="#444", sw=1.5):
        if rw > 1 and rh > 1:
            el.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" '
                      f'fill="none" stroke="{stroke}" stroke-width="{sw}"/>')

    el.append(f'<rect width="{canvas_w}" height="{canvas_h}" fill="white"/>')
    R(x0, y0, furn_w, furn_h, "#1a1a1a", 3.0)

    if has_plinth:
        ph = max(6, round(furn_h * 0.03))
        R(x0, y1, furn_w, ph, "#333", 1.5)

    for sec in sections:
        sx0_pct = float(sec.get("x_start_pct", 0))
        sx1_pct = float(sec.get("x_end_pct",  100))
        sx0 = xp(sx0_pct)
        sx1 = xp(sx1_pct)
        sw  = sx1 - sx0
        t   = sec.get("type", "").lower()

        # structural divider
        if sx0_pct > 0.5:
            L(sx0, y0, sx0, y1, "#1a1a1a", 2.0)

        # shelves
        for shelf in sec.get("shelves", []):
            sy   = yp(shelf["y_pct"])
            shx0 = xp(shelf.get("x_start_pct", sx0_pct))
            shx1 = xp(shelf.get("x_end_pct",   sx1_pct))
            L(shx0, sy, shx1, sy, "#444", 1.5)

        # slot dividers
        for div in sec.get("slot_dividers", []):
            dx  = xp(div["x_pct"])
            dy0 = yp(div.get("y_start_pct", 0))
            dy1 = yp(div.get("y_end_pct",  100))
            L(dx, dy0, dx, dy1, "#666", 1.2)

        # closed doors
        dc = int(sec.get("doors", 0))
        if dc > 0 and "door" in t:
            for d in range(dc):
                dsx0 = sx0 + round(sw * d / dc)
                dsx1 = sx0 + round(sw * (d + 1) / dc)
                dsw  = dsx1 - dsx0
                if d > 0:
                    L(dsx0, y0, dsx0, y1, "#444", 1.5)
                R(dsx0 + _PADDING, y0 + _PADDING,
                  dsw - 2*_PADDING, furn_h - 2*_PADDING, "#444", 1.2)

        # drawers
        dr = int(sec.get("drawers", 0))
        if dr > 0:
            full_shelves = [
                s for s in sec.get("shelves", [])
                if abs(float(s.get("x_start_pct", 0))   - sx0_pct) < 2
                and abs(float(s.get("x_end_pct", 100)) - sx1_pct) < 2
            ]
            drz0  = yp(min(float(s["y_pct"]) for s in full_shelves)) if full_shelves else y0
            drz1  = y1
            drz_h = drz1 - drz0
            for d in range(1, dr):
                dy = drz0 + round(drz_h * d / dr)
                L(sx0, dy, sx1, dy, "#444", 1.5)
            for d in range(dr):
                ry0 = drz0 + round(drz_h * d / dr)     + _PADDING
                ry1 = drz0 + round(drz_h * (d+1) / dr) - _PADDING
                R(sx0 + _PADDING, ry0, sw - 2*_PADDING, ry1 - ry0, "#444", 1.2)

        # label
        mid_x = (sx0 + sx1) // 2
        sid   = sec.get("section_id", "")
        w_pct = round(sx1_pct - sx0_pct, 1)
        el.append(f'<text x="{mid_x}" y="{canvas_h - 4}" text-anchor="middle" '
                  f'font-family="Arial,sans-serif" font-size="9" fill="#888">'
                  f'{sid} {w_pct}%</text>')

    inner = "\n  ".join(el)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {canvas_w} {canvas_h}" '
            f'width="{canvas_w}" height="{canvas_h}">\n  {inner}\n</svg>')


# ==============================================================
#  MAIN ENTRY POINT
# ==============================================================

class SvgGenerator:

    def run(self,
            image_bgr:     np.ndarray,
            wireframe_img: np.ndarray) -> str:
        """
        image_bgr     — original furniture photo (for model classification)
        wireframe_img — your OpenCV wireframe output (for Python line detection)
        """

        # Step 1 — Python detects line positions from wireframe
        line_data = detect_structural_lines(wireframe_img)

        # Step 2 — Model classifies content (no measuring)
        structure = classify_sections(image_bgr, line_data)

        # Step 3 — Python computes slot positions
        structure = _expand_slots(structure)

        # Step 4 — Python builds SVG
        svg = build_svg(structure)

        print(f"  [svg] built {len(svg)} chars, "
              f"{len(structure.get('sections', []))} sections")
        return svg
