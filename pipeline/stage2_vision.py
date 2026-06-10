"""
Stage 2 — Vision Analysis
Sends the cropped furniture image to GPT-4.1 vision and returns a
structured JSON dict describing every visible compartment, drawer, door,
shelf, and divider.
"""

import base64
import json

import cv2
import numpy as np
from openai import OpenAI

_PROMPT = """Analyze this furniture image with extreme precision.

Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

Use this exact structure:
{
  "furniture_type": "overall type e.g. modular storage unit, wardrobe, sideboard",
  "perspective": "e.g. slightly angled front view, straight on",
  "proportions": {
    "width_to_height": 3.2,
    "depth_notes": "shallow wall-mounted unit"
  },
  "sections": [
    {
      "name": "left cabinet",
      "position": "left",
      "type": "two-door cabinet",
      "width_fraction": 0.15,
      "height_fraction": 1.0,
      "details": "two full-height doors, no handles",
      "counts": {
        "doors": 2,
        "drawers": 0,
        "shelves": 0,
        "vertical_dividers": 1,
        "horizontal_dividers": 0,
        "compartments": 1
      },
      "sub_sections": []
    }
  ],
  "drawers": ["describe each drawer or drawer group"],
  "doors": [
    {
      "description": "one full-height door with slanted top",
      "handle_type": "vertical_pull_bar | horizontal_bar | round_knob | oval_loop | d_ring | recessed | none",
      "handle_position": "left_center | right_center | top_center | center"
    }
  ],
  "shelves": ["describe each visible shelf row"],
  "vertical_dividers": ["describe each vertical partition panel"],
  "horizontal_dividers": ["describe each horizontal partition panel"],
  "total_counts": {
    "doors": 4,
    "drawers": 3,
    "sections": 8
  }
}

Rules:
- width_fraction: what fraction of the total unit width this section occupies (all sections must sum to 1.0)
- height_fraction: what fraction of the total unit height this section spans (1.0 = full height)
- counts: exact integer counts of every element inside this section — count carefully
- List every section from left to right
- Do not group sections together — each visually distinct column or zone is its own section
- Be precise about compartment counts. If you see 6 vertical slots, write 6, not "several"
- For doors: identify the exact handle type from the list — vertical_pull_bar is a thin straight bar, round_knob is a circular protrusion, oval_loop is an elliptical loop, recessed is a finger groove with no protruding hardware
- total_counts must reflect the entire unit

Return ONLY the JSON object with no surrounding text."""


def analyze_vision(image_bgr: np.ndarray, client: OpenAI) -> dict:
    b64  = _encode_png(image_bgr)
    resp = client.chat.completions.create(
        model="gpt-4.1",
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {
                    "url":    f"data:image/png;base64,{b64}",
                    "detail": "high",
                }},
            ],
        }],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Strip accidental markdown fences if the model added them
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)


def _encode_png(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", image_bgr)
    return base64.b64encode(buf).decode("utf-8")
