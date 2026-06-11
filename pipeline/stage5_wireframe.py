"""
Stage 5 — Wireframe Generation (gpt-image-2, trace-and-clean mode)

The edge map from Stage 4 is already structurally correct.
The model's only job is to redraw those lines cleanly — not to
redesign, interpret, or add anything.

Two images are sent:
  IMAGE 1 — cropped furniture photo  (scale / proportion reference only)
  IMAGE 2 — edge map                 (the structural template to trace exactly)
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI

_MAX_SIDE = 1024
_PADDING  = 0.50   # white margin added around the image before sending

_PROMPT = """\
You are given two images:
  IMAGE 1 — a furniture photograph (reference for scale and proportion only)
  IMAGE 2 — an edge map of that furniture (black lines on white background)

IMAGE 2 is already the correct wireframe. Your job is ONLY to redraw its \
lines as clean, uniform technical lines.

RULES:
1. Every black line visible in IMAGE 2 must appear in your output at the same position.
2. Do NOT add any lines that are not in IMAGE 2.
3. Do NOT remove or skip any lines that are in IMAGE 2.
4. Do NOT merge adjacent compartments or simplify groups of lines.
5. Do NOT change proportions, section widths, or compartment counts.
6. Draw handles exactly as they appear in IMAGE 2 — a short vertical stroke \
stays a short vertical stroke; do not turn it into a circle or oval.
7. Preserve all diagonal lines (e.g. slanted cabinet tops) exactly as angled.
8. If a line in IMAGE 2 almost reaches a corner or endpoint but stops \
slightly short, extend it to close the gap. Do not add new lines — only \
extend existing ones to their natural endpoint.

OUTPUT:
- Pure white background (#FFFFFF)
- Clean sharp black strokes (#000000), uniform 1–2px weight
- No texture, color, shading, shadows, floor, wall, or background
- No labels or annotations
"""


def generate_wireframe(
    cropped_bgr: np.ndarray,
    edge_map_bgr: np.ndarray,
    description: str,
    client: OpenAI,
    vision_json: dict | None = None,
) -> np.ndarray:
    orig_bytes = _to_png_bytes(_resize(_pad(cropped_bgr)))
    edge_bytes = _to_png_bytes(_resize(_pad(edge_map_bgr)))

    response = client.images.edit(
        model="gpt-image-1",
        image=[
            ("original.png", orig_bytes, "image/png"),
            ("edges.png",    edge_bytes, "image/png"),
        ],
        prompt=_PROMPT,
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


def _pad(image_bgr: np.ndarray, fraction: float = _PADDING) -> np.ndarray:
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
    return cv2.resize(
        image_bgr, (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _to_png_bytes(image_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", image_bgr)
    return bytes(buf)
