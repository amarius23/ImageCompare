from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class WireframeResult:
    clean_img: object    # np.ndarray uint8 — white bg, black lines
    soft_map:  object    # np.ndarray float32 — blurred for comparison
    aspect_ratio: float = 1.0
    svg_string: str = ""  # raw SVG source when GPT-based; empty for CV-based

    def to_dict(self) -> dict:
        return {"aspect_ratio": round(self.aspect_ratio, 4)}


@dataclass
class ComparisonDifference:
    type: str
    description: str
    severity: str        # "critical" | "warning"
    original: Any = None
    generated: Any = None


@dataclass
class ComparisonReport:
    verdict: str         # "PASS" | "FAIL"
    confidence: float
    differences: List[ComparisonDifference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 1),
            "differences": [
                {
                    "type": d.type,
                    "description": d.description,
                    "severity": d.severity,
                    "original": d.original,
                    "generated": d.generated,
                }
                for d in self.differences
            ],
        }
