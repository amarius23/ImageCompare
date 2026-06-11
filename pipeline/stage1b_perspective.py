"""
Stage 1b — Perspective Check
Detects whether the furniture is photographed straight-on (frontal, one face
visible) or at an angle (3/4 / side view, two or more faces visible).

A frontal shot shows only the flat front panel.
An angled shot shows the front panel PLUS a visible side panel, creating
a clear 3D wedge or parallelogram outline (as in cabinet_01 original).

Result is saved as 1b_perspective.json and passed to Stage 6 so that
perspective-mismatched pairs are automatically flagged as NO MATCH.
"""

import base64
import json

import cv2
import numpy as np
from openai import OpenAI

_PROMPT = """\
Look at this furniture photograph and determine the camera perspective.

Return ONLY a valid JSON object — no markdown, no explanation.

JSON structure:
{
  "is_frontal": true or false,
  "view_type": "frontal" | "three_quarter" | "side" | "top_down",
  "confidence": 0-100,
  "reasoning": "one sentence explaining what you see"
}

Definitions:
  frontal      — camera faces the furniture straight-on; only the front face is
                 visible; left and right edges are vertical parallel lines.
  three_quarter — camera is at a diagonal; the front face AND a side face are
                 both clearly visible; the top or bottom edge is a slanting line.
  side         — only the side panel is visible; the front face is hidden.
  top_down     — camera looks downward at the furniture from above.

Focus only on the main furniture piece, not the room or background.
"""


def check_perspective(cropped_bgr: np.ndarray, client: OpenAI) -> dict:
    _, buf = cv2.imencode(".png", cropped_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")

    resp = client.chat.completions.create(
        model="gpt-4.1",
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}", "detail": "low",
                }},
            ],
        }],
    )

    raw = resp.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)

    view     = result.get("view_type",  "unknown")
    is_front = result.get("is_frontal", False)
    conf     = result.get("confidence", 0)
    reason   = result.get("reasoning",  "")
    tag      = "FRONTAL" if is_front else f"ANGLED ({view})"
    print(f"    [S1b] {tag}  confidence={conf}%")
    print(f"          {reason}")
    return result
