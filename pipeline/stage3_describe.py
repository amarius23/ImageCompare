"""
Stage 3 — Description Builder
Converts the vision JSON from Stage 2 into a concise structured text
description that guides the wireframe generation model in Stage 5.
"""


def build_description(vision_json: dict) -> str:
    lines: list[str] = []

    furniture_type = vision_json.get("furniture_type", "furniture piece")
    perspective    = vision_json.get("perspective", "")
    proportions    = vision_json.get("proportions", {})

    lines.append(f"Furniture: {furniture_type}.")

    if perspective:
        lines.append(f"Perspective: {perspective}.")

    if proportions:
        w_h = proportions.get("width_to_height")
        if w_h:
            lines.append(f"Width-to-height ratio: approximately {w_h}.")
        depth = proportions.get("depth_notes")
        if depth:
            lines.append(f"Depth: {depth}.")

    total = vision_json.get("total_counts", {})
    if total:
        parts = []
        if total.get("doors"):    parts.append(f"{total['doors']} doors total")
        if total.get("drawers"):  parts.append(f"{total['drawers']} drawers total")
        if total.get("sections"): parts.append(f"{total['sections']} sections")
        if parts:
            lines.append(f"Total: {', '.join(parts)}.")

    sections = vision_json.get("sections", [])
    if sections:
        lines.append(f"\nSections ({len(sections)} total, left to right):")
        cursor = 0.0
        for i, sec in enumerate(sections, 1):
            name     = sec.get("name", "section")
            sec_type = sec.get("type", "")
            details  = sec.get("details", "")
            w_frac   = sec.get("width_fraction") or 0.0
            h_frac   = sec.get("height_fraction")
            counts   = sec.get("counts", {})

            x_start = int(round(cursor * 100))
            x_end   = int(round((cursor + w_frac) * 100))
            cursor  += w_frac

            header = f"  [{i}] {name}: {sec_type}."
            header += f"  X: {x_start}%–{x_end}% of total width."
            if h_frac and h_frac < 1.0:
                header += f"  Height span: {int(h_frac * 100)}%."
            if details:
                header += f"  {details}"
            lines.append(header)

            if counts:
                count_parts = []
                for k, v in counts.items():
                    if v and v > 0:
                        count_parts.append(f"{v} {k.replace('_', ' ')}")
                if count_parts:
                    lines.append(f"       EXACT COUNTS: {', '.join(count_parts)}")

            for sub in sec.get("sub_sections", []):
                lines.append(f"    └ {sub}")

    _append_list(lines, "Drawers",             vision_json.get("drawers",             []))
    _append_list(lines, "Doors",               vision_json.get("doors",               []))
    _append_list(lines, "Shelves",             vision_json.get("shelves",             []))
    _append_list(lines, "Vertical dividers",   vision_json.get("vertical_dividers",   []))
    _append_list(lines, "Horizontal dividers", vision_json.get("horizontal_dividers", []))

    lines.append("\nAll geometry consists of rectangular panels. Preserve exact proportions.")

    return "\n".join(lines)


def _append_list(lines: list[str], label: str, items: list) -> None:
    if items:
        lines.append(f"\n{label}:")
        for item in items:
            lines.append(f"  - {item}")
