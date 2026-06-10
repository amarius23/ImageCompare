"""
Stage 5 — Wireframe Generation
Uses gpt-image-1 image editing to transform the furniture photograph
into a clean technical wireframe.

Two images are sent as visual references:
  1. The cropped furniture photo  (original appearance + proportions)
  2. The edge map from Stage 4    (structural boundary guide)

The structural description from Stage 3 is embedded in the prompt.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

_PROMPT_TEMPLATE = """\
You are given two reference images:
  IMAGE 1 — the cropped furniture photograph (use for overall proportions and depth cues)
  IMAGE 2 — the edge map (black lines on white, use as the PRIMARY tracing template)

Your task: produce a clean technical wireframe of this furniture by tracing IMAGE 2 as precisely as possible.

Structural layout (left to right):
{description}{json_block}

DRAWING INSTRUCTIONS — follow in order:

Step 1 — Outer frame
  Draw the exact outer bounding rectangle of the unit using the x/height proportions above.

Step 2 — Section boundaries
  For each section, draw a vertical dividing line at its X% position from the left edge.
  The X-start and X-end positions are given for every section — place lines there exactly.

Step 3 — Internal structure (section by section)
  For each section, draw every internal horizontal shelf, vertical divider, and drawer line.
  Use the EXACT COUNTS: if a section says "6 compartments / 5 vertical dividers", draw exactly 5 lines.
  Refer to IMAGE 2 to confirm the position and spacing of each internal line.

Step 4 — Handles and hardware
  Draw each handle using its exact type:
    vertical_pull_bar  → a short thin vertical straight line
    horizontal_bar     → a short thin horizontal straight line
    round_knob         → a small circle
    oval_loop          → a small ellipse
    recessed / none    → nothing
  Position the handle at the stated location (left_center, right_center, etc.).

STRICT RULES:
- Do not invent or omit any section, compartment, shelf, or divider
- Do not merge compartments together or split one into two
- Section widths must match their stated X% ranges — do not compress or stretch any zone
- Do not substitute handle shapes
- If IMAGE 2 shows a line, draw it. If IMAGE 2 does not show a line, do not draw it.

OUTPUT:
- White background
- Clean black lines, uniform 1–2px stroke weight
- No texture, color, shading, shadows, floor, wall, or background
- No labels or annotations\
"""

# Resize before sending so we stay within API limits and cost bounds
_MAX_SIDE = 1024


def generate_wireframe(
    cropped_bgr: np.ndarray,
    edge_map_bgr: np.ndarray,
    description: str,
    client: OpenAI,
    vision_json: dict | None = None,
) -> np.ndarray:
    orig_bytes = _to_png_bytes(_resize(_pad(cropped_bgr)))
    edge_bytes = _to_png_bytes(_resize(_pad(edge_map_bgr)))

    json_block = ""
    if vision_json:
        import json as _json
        json_block = f"\n\nMachine-readable structure (use for exact counts):\n{_json.dumps(vision_json, indent=2)}"

    prompt = _PROMPT_TEMPLATE.format(description=description, json_block=json_block)

    response = client.images.edit(
        model="gpt-image-2",
        image=[
            ("original.png", orig_bytes, "image/png"),
            ("edges.png",    edge_bytes, "image/png"),
        ],
        prompt=prompt,
        size="1024x1024",
        quality="high",
    )

    item = response.data[0]
    if item.b64_json:
        img_bytes = base64.b64decode(item.b64_json)
    else:
        import urllib.request
        img_bytes = urllib.request.urlopen(item.url).read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _pad(image_bgr: np.ndarray, fraction: float = 0.80) -> np.ndarray:
    h, w  = image_bgr.shape[:2]
    pad_h = int(h * fraction)
    pad_w = int(w * fraction)
    return cv2.copyMakeBorder(
        
        
        image_bgr, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def _resize(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if max(h, w) <= _MAX_SIDE:
        return image_bgr
    scale = _MAX_SIDE / max(h, w)
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _to_png_bytes(image_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", image_bgr)
    return bytes(buf)
