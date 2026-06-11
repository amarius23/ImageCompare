"""
Stage 6 — Wireframe Comparison
Compares the original wireframe against the generated wireframe and
returns a structured verdict with a confidence score and specific
differences / matching elements.

Two methods are combined:
  • SSIM  — fast pixel-level structural similarity score (0–1)
  • GPT-4.1 vision — semantic structural comparison (sections, counts,
    proportions, compartment layout)
"""

import base64
import json

import cv2
import numpy as np
from openai import OpenAI

try:
    from skimage.metrics import structural_similarity as ssim
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False

_COMPARE_PROMPT = """\
You are comparing two furniture wireframes.
WIREFRAME A is the ORIGINAL (reference).
WIREFRAME B is the GENERATED (being evaluated).

Both images show the same furniture model as clean black line drawings on \
a white background.

Analyze both wireframes and return ONLY a valid JSON object — no markdown, \
no explanation.

JSON structure:
{
  "verdict": "MATCH" or "NO MATCH",
  "confidence": 0-100,
  "original_structure": {
    "section_count": int,
    "total_doors": int,
    "total_drawers": int,
    "total_open_bays": int,
    "vertical_dividers": int,
    "horizontal_dividers": int,
    "width_to_height_ratio": float
  },
  "generated_structure": {
    "section_count": int,
    "total_doors": int,
    "total_drawers": int,
    "total_open_bays": int,
    "vertical_dividers": int,
    "horizontal_dividers": int,
    "width_to_height_ratio": float
  },
  "matching_elements": ["list of structural elements that match correctly"],
  "differences": ["list of specific structural differences found"],
  "summary": "one sentence overall assessment"
}

Confidence scoring:
  90-100: structures are identical or near-identical
  70-89:  minor differences, 1-2 small elements wrong
  50-69:  moderate differences, section counts match but content differs
  30-49:  significant differences, wrong section count or major mismatch
  0-29:   completely different structures

Verdict rules:
  MATCH    — confidence >= 70 AND section_count matches AND \
overall proportions are correct
  NO MATCH — everything else
"""


def compare_wireframes(
    orig_wf_bgr: np.ndarray,
    gen_wf_bgr: np.ndarray,
    client: OpenAI,
) -> dict:
    # ── SSIM ──────────────────────────────────────────────────────────────────
    ssim_score: float | None = None
    if _SSIM_AVAILABLE:
        target = (512, 512)
        a = cv2.resize(orig_wf_bgr, target)
        b = cv2.resize(gen_wf_bgr,  target)
        a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        ssim_score = float(ssim(a_g, b_g, win_size=7))

    # ── GPT-4.1 vision comparison ──────────────────────────────────────────────
    a_b64 = _encode(orig_wf_bgr)
    b_b64 = _encode(gen_wf_bgr)

    resp = client.chat.completions.create(
        model="gpt-4.1",
        max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _COMPARE_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{a_b64}", "detail": "high",
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b_b64}", "detail": "high",
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

    if ssim_score is not None:
        result["ssim_score"] = round(ssim_score, 4)

    verdict    = result.get("verdict",    "ERROR")
    confidence = result.get("confidence", 0)
    summary    = result.get("summary",    "")
    ssim_str = f"{ssim_score:.3f}" if ssim_score is not None else "n/a"
    print(f"    [S6] {verdict}  confidence={confidence}%  ssim={ssim_str}")
    print(f"         {summary}")

    diffs = result.get("differences", [])
    for d in diffs:
        print(f"         ✗ {d}")
    matches = result.get("matching_elements", [])
    for m in matches[:3]:
        print(f"         ✓ {m}")

    return result


def _encode(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buf).decode("utf-8")
