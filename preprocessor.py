import os
import cv2
import numpy as np
import base64
import json

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR  = os.path.join(BASE_DIR, "logs", "crops")
FRAMES_DIR = os.path.join(BASE_DIR, "logs", "frames")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4o"

COMPARISON_PROMPT = (
    "You are a senior furniture quality assurance inspector with expertise in structural analysis. "
    "You will be shown two furniture images. IMAGE 1 is the ORIGINAL furniture which is the reference. "
    "IMAGE 2 is the GENERATED furniture which is the one being evaluated. "
    "Your job is to compare their STRUCTURE only. "
    "Ignore color, material finish, wood grain, lighting, shadows, room background, and camera angle differences. "
    "Focus only on structural layout, section count, door count, drawer count, shelf positions, and proportions.\n\n"
    "Analyze IMAGE 1 first and note the overall shape including whether it is wide, tall, or square and its "
    "approximate ratio, how many vertical sections exist, what is in each section including doors, open bays, "
    "drawers, shelves, and slot dividers, how many doors total, how many drawers total, and whether there are "
    "any special features like diagonal sections or slanted tops.\n\n"
    "Then analyze IMAGE 2 the same way.\n\n"
    "Then compare them and return ONLY this JSON object with no markdown and no explanation. "
    "The JSON must have these fields. "
    "verdict which is either the string MATCH or the string NO MATCH. "
    "confidence which is an integer from 0 to 100. "
    "original_structure which is an object containing width_to_height_ratio as a float, "
    "section_count as an integer, total_doors as an integer, total_drawers as an integer, "
    "total_open_bays as an integer, has_diagonal_or_irregular as a boolean, and sections as an array "
    "where each element has position as one of left or center-left or center or center-right or right, "
    "type as one of closed_door or open_bay or drawers or mixed or irregular, "
    "doors as an integer, drawers as an integer, shelves as an integer, and slot_dividers as an integer. "
    "generated_structure which has the exact same schema as original_structure. "
    "differences which is an array of strings where each string describes one specific structural difference found. "
    "matching_elements which is an array of strings where each string describes one structural element that matches correctly. "
    "section_count_match as a boolean. "
    "door_count_match as a boolean. "
    "drawer_count_match as a boolean. "
    "overall_proportion_match as a boolean. "
    "summary as a single sentence describing the overall comparison result.\n\n"
    "For the confidence scoring use these rules. "
    "90 to 100 means structures are identical or near identical. "
    "70 to 89 means minor differences with one or two small elements wrong. "
    "50 to 69 means moderate differences where section counts match but content differs. "
    "30 to 49 means significant differences with wrong section count or major content mismatch. "
    "0 to 29 means completely different structures.\n\n"
    "For the verdict use these rules. "
    "Return MATCH only if confidence is 70 or above AND section_count_match is true AND door_count_match is true. "
    "Return NO MATCH for everything else."
)


# ==============================================================
#  UTILITIES
# ==============================================================

def _ensure_dirs():
    os.makedirs(CROPS_DIR,  exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)


def _encode_jpg(image_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


# ==============================================================
#  LOCALIZATION — rembg background removal → tight crop
# ==============================================================

def crop_to_furniture(image_path: str, padding: int = 20) -> np.ndarray:
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
        print(f"  [rembg] failed ({e}) — using full image.")
        return img


# ==============================================================
#  DIRECT VISUAL COMPARISON
# ==============================================================

def compare_furniture(original_bgr: np.ndarray, generated_bgr: np.ndarray) -> dict:
    from openai import OpenAI
    client   = OpenAI(api_key=OPENAI_API_KEY)
    orig_b64 = _encode_jpg(original_bgr)
    gen_b64  = _encode_jpg(generated_bgr)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text",
             "text": COMPARISON_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}", "detail": "high"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{gen_b64}",  "detail": "high"}},
        ]}]
    )

    raw = resp.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
    except Exception:
        print(f"  [compare] JSON parse failed. Raw: {raw[:200]}")
        return {"verdict": "ERROR", "confidence": 0}

    verdict    = result.get("verdict",    "NO MATCH")
    confidence = result.get("confidence", 0)
    summary    = result.get("summary",    "")
    print(f"  [compare] {verdict}  confidence={confidence}%  — {summary}")
    return result


# ==============================================================
#  PIPELINE
# ==============================================================

def run_qa(original_path: str, generated_path: str, auto_bbox: bool = True) -> dict:
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

    result  = compare_furniture(img_orig, img_gen)
    verdict = result.get("verdict", "ERROR")

    out_path = os.path.join(FRAMES_DIR, f"{image_name}_comparison_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return {
        "image_name":  image_name,
        "orig_crop":   orig_crop,
        "gen_crop":    gen_crop,
        "verdict":     verdict,
        "match":       verdict == "MATCH",
        "result_path": out_path,
    }


def run_batch(pairs: list) -> list:
    results = []
    for pair in pairs:
        r = run_qa(pair["original"], pair["generated"],
                   auto_bbox=pair.get("auto_bbox", True))
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
    # {"name": "cabinet_01", "original": "images/originals/5.png",  "generated": "images/generated/5.png",  "auto_bbox": True},
    # {"name": "cabinet_02", "original": "images/originals/1.jpg",  "generated": "images/generated/1.jpg",  "auto_bbox": True},
    {"name": "cabinet_03", "original": "images/originals/2.jpg",  "generated": "images/generated/2.jpg",  "auto_bbox": True},
    {"name": "cabinet_04", "original": "images/originals/3.jpg",  "generated": "images/generated/3.jpg",  "auto_bbox": True},
    # {"name": "cabinet_05", "original": "images/originals/4.jpg",  "generated": "images/generated/4.jpg",  "auto_bbox": True},
    # {"name": "cabinet_06", "original": "images/originals/6.jpg",  "generated": "images/generated/6.jpg",  "auto_bbox": True},
    # {"name": "cabinet_07", "original": "images/originals/7.jpg",  "generated": "images/generated/7.jpg",  "auto_bbox": True},
    {"name": "cabinet_08", "original": "images/originals/8.jpg",  "generated": "images/generated/8.jpg",  "auto_bbox": True},
    {"name": "cabinet_09", "original": "images/originals/9.jpg",  "generated": "images/generated/9.jpg",  "auto_bbox": True},
    # {"name": "cabinet_10", "original": "images/originals/10.png", "generated": "images/generated/10.png", "auto_bbox": True},
    # {"name": "cabinet_11", "original": "images/originals/11.png", "generated": "images/generated/11.png", "auto_bbox": True},
]

if __name__ == "__main__":
    run_batch(IMAGE_PAIRS)
