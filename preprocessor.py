# ============================================================
#  qa_tool.py  —  Furniture QA: Image Structural Integrity Checker
#  Usage: python qa_tool.py  |  Add image pairs to IMAGE_PAIRS below.
# ============================================================

import os
import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


# ==============================================================
#  SECTION 1 — CONSTANTS
# ==============================================================

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ORIGINALS_DIR  = os.path.join(BASE_DIR, "images", "originals")
GENERATED_DIR  = os.path.join(BASE_DIR, "images", "generated")
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
DIFF_MAPS_DIR  = os.path.join(BASE_DIR, "logs", "diff_maps")
LOG_FILE       = os.path.join(BASE_DIR, "logs", "qa_results.log")

TARGET_WIDTH         = 512
TARGET_HEIGHT        = 512
CONVERT_TO_GRAYSCALE = True

SSIM_STRICT_MODE = True   # True → 0.85 threshold; False → 0.75 (lenient for different-room shots)
SSIM_THRESHOLD   = 0.85 if SSIM_STRICT_MODE else 0.75
SSIM_WIN_SIZE    = 7
SAVE_DIFF_MAP    = True

FSIM_ENABLED   = False
FSIM_THRESHOLD = 0.80

# Minimum LAB L* std-dev required to trust gradient-based bbox detection.
AUTO_BBOX_MIN_CONTRAST = 20.0

OPENAI_BBOX_ENABLED = True
OPENAI_API_KEY      = ""
OPENAI_BBOX_MODEL   = "gpt-4o"     # change to gpt-4o-mini to reduce cost

REMBG_ENABLED = True

LOFTR_ENABLED    = True
LOFTR_PRETRAINED = "indoor"

LPIPS_ENABLED   = True
LPIPS_NET       = "alex"
LPIPS_THRESHOLD = 0.45     # distance; 0=identical, 1=completely different

HASH_ENABLED   = True
HASH_SIZE      = 16
HASH_THRESHOLD = 15        # Hamming distance; out of HASH_SIZE²=256 bits

LOG_TO_FILE = True
LOG_LEVEL   = "INFO"


# ==============================================================
#  SECTION 2 — IMAGE ALIGNMENT
# ==============================================================

def _is_homography_sane(H, src_w, src_h, dst_w, dst_h, inliers, good_match_count):
    """
    Rejects homographies caused by ORB matching background objects (plants, lamps)
    instead of the furniture. Those matches can be internally consistent yet produce
    catastrophic warps — zoom to a plant, collapse to a point, rotate 90°.

    Checks: corner blow-out, area collapse, scale extremes, low inlier ratio,
    excessive rotation (>35°). Returns (True, "") or (False, reason).
    """
    corners = np.float32([[0,0],[src_w,0],[src_w,src_h],[0,src_h]]).reshape(-1,1,2)
    mapped  = cv2.perspectiveTransform(corners, H)
    mx, my  = dst_w * 0.5, dst_h * 0.5
    for cx, cy in mapped.reshape(-1, 2):
        if not (-mx <= cx <= dst_w + mx):
            return False, f"corner x={cx:.0f} outside allowed range"
        if not (-my <= cy <= dst_h + my):
            return False, f"corner y={cy:.0f} outside allowed range"

    hull_area = cv2.contourArea(mapped.reshape(4, 2).astype(np.float32))
    if hull_area < dst_w * dst_h * 0.10:
        return False, f"mapped area {hull_area:.0f}px² < 10% of canvas"

    scale = np.linalg.det(H[:2, :2])
    if not (0.5 <= scale <= 2.0):
        return False, f"scale det {scale:.4f} outside [0.5, 2.0]"

    # Bad warps observed had scale_det 0.286–0.485 (background-depth mismatch);
    # legitimate camera corrections produced ~1.0.
    inlier_ratio = inliers / good_match_count if good_match_count > 0 else 0.0
    if inlier_ratio < 0.30:
        return False, f"inlier ratio {inlier_ratio:.2f} ({inliers}/{good_match_count}) < 0.30"

    U, _, Vt = np.linalg.svd(H[:2, :2])
    R = U @ Vt
    rot = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    if rot > 35.0:
        return False, f"rotation {rot:.1f}° > 35°"

    return True, ""


def _try_loftr_alignment(img_original, img_generated):
    """
    Aligns img_generated to img_original using LoFTR (detector-free transformer).

    LoFTR beats ORB on plain furniture: it matches dense patches rather than requiring
    salient keypoints, which plain white wardrobes almost entirely lack.
    Returns warped image or None (caller falls back to ORB).
    """
    try:
        import torch
        import kornia.feature as KF
    except ImportError:
        print("  [LoFTR] kornia not installed — falling back to ORB.")
        return None

    oh, ow = img_original.shape[:2]
    LW, LH = 640, 480

    def _prep(bgr):
        gray = cv2.cvtColor(cv2.resize(bgr, (LW, LH)), cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]

    try:
        matcher = KF.LoFTR(pretrained=LOFTR_PRETRAINED)
        matcher.eval()
        with torch.no_grad():
            out = matcher({"image0": _prep(img_original), "image1": _prep(img_generated)})
    except Exception as e:
        print(f"  [LoFTR] Inference error: {e} — falling back to ORB.")
        return None

    kp0  = out["keypoints0"].cpu().numpy()
    kp1  = out["keypoints1"].cpu().numpy()
    mask = out["confidence"].cpu().numpy() >= 0.5
    kp0, kp1 = kp0[mask], kp1[mask]
    n = len(kp0)
    print(f"  [LoFTR] {n} high-confidence matches.")

    if n < 20:
        print("  [LoFTR] Too few matches — falling back to ORB.")
        return None

    gh, gw = img_generated.shape[:2]
    src = (kp1 * [gw/LW, gh/LH]).reshape(-1,1,2).astype(np.float32)
    dst = (kp0 * [ow/LW, oh/LH]).reshape(-1,1,2).astype(np.float32)

    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("  [LoFTR] Homography returned None — falling back to ORB.")
        return None

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    ratio   = inliers / n
    print(f"  [LoFTR] RANSAC: {inliers}/{n} inliers (ratio {ratio:.2f}).")

    # LoFTR produces semi-dense matches so its inlier ratio is naturally lower than
    # ORB's. 15% derived empirically: ambiguous-panel warps scored 14%, good pairs ≥15%.
    if ratio < 0.15:
        print(f"  [LoFTR] Ratio {ratio:.2f} < 0.15 — falling back to ORB.")
        return None

    sane, reason = _is_homography_sane(H, gw, gh, ow, oh, inliers, inliers)
    if not sane:
        print(f"  [LoFTR] Sanity check failed ({reason}) — falling back to ORB.")
        return None

    print("  [LoFTR] Homography OK — applying warp.")
    return cv2.warpPerspective(img_generated, H, (ow, oh))


def align_images(img_original, img_generated):
    """
    Warps img_generated onto img_original's perspective via LoFTR → ORB → resize fallback.
    Runs before bbox crop so the original's bbox correctly addresses the same furniture
    region in the generated image.
    """
    MIN_MATCHES = 25
    RATIO       = 0.70
    MIN_INLIERS = 15

    oh, ow = img_original.shape[:2]
    gh, gw = img_generated.shape[:2]
    fallback = cv2.resize(img_generated, (ow, oh))

    if LOFTR_ENABLED:
        result = _try_loftr_alignment(img_original, img_generated)
        if result is not None:
            return result
        print("  [align] LoFTR failed — trying ORB.")

    gray_orig = cv2.cvtColor(img_original,  cv2.COLOR_BGR2GRAY)
    gray_gen  = cv2.cvtColor(img_generated, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_orig, desc_orig = orb.detectAndCompute(gray_orig, None)
    kp_gen,  desc_gen  = orb.detectAndCompute(gray_gen,  None)

    if desc_orig is None or desc_gen is None:
        print("  [align] No descriptors — skipping alignment.")
        return fallback

    bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    good = []
    for pair in bf.knnMatch(desc_gen, desc_orig, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            good.append(pair[0])
    print(f"  [align] {len(good)} good matches (need ≥{MIN_MATCHES}).")

    if len(good) < MIN_MATCHES:
        print("  [align] Too few matches — skipping alignment.")
        return fallback

    src = np.float32([kp_gen[m.queryIdx].pt  for m in good]).reshape(-1,1,2)
    dst = np.float32([kp_orig[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("  [align] Homography returned None — skipping alignment.")
        return fallback

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"  [align] RANSAC: {inliers}/{len(good)} inliers.")

    if inliers < MIN_INLIERS:
        print(f"  [align] Only {inliers} inliers — skipping alignment.")
        return fallback

    sane, reason = _is_homography_sane(H, gw, gh, ow, oh, inliers, len(good))
    if not sane:
        print(f"  [align] Sanity check failed ({reason}) — skipping alignment.")
        return fallback

    print("  [align] Homography OK — applying warp.")
    return cv2.warpPerspective(img_generated, H, (ow, oh))


# ==============================================================
#  SECTION 3 — PREPROCESSING
# ==============================================================

def load_image(image_path):
    """Loads image from disk; raises FileNotFoundError or ValueError on failure."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img


def _resize_with_pad(image, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT,
                     pad_val=127, interpolation=None):
    """
    Scales image to fit within target_w×target_h (aspect-ratio preserved), then pads.
    pad_val=127 (mid-grey) produces zero Canny edges at the border — no spurious SSIM contribution.
    """
    h, w  = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    canvas  = np.full(
        (target_h, target_w, image.shape[2]) if image.ndim == 3 else (target_h, target_w),
        pad_val, dtype=np.uint8,
    )
    y0, x0 = (target_h - nh) // 2, (target_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def resize_image(image):
    if image.shape[0] == TARGET_HEIGHT and image.shape[1] == TARGET_WIDTH:
        return image
    return _resize_with_pad(image, TARGET_WIDTH, TARGET_HEIGHT, pad_val=127)


def to_grayscale(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def align_crops_ecc(img_orig_gray, img_gen_gray):
    """
    Uses ECC (Enhanced Correlation Coefficient) to find the optimal
    translation+scale transform that aligns img_gen to img_orig.
    ECC is ideal here: it handles pure translation and zoom differences
    (exactly what we have) and is robust to lighting/contrast differences.
    Returns aligned img_gen_gray, same shape as input.
    """
    warp_mode   = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            img_orig_gray.astype(np.float32),
            img_gen_gray.astype(np.float32),
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        aligned = cv2.warpAffine(
            img_gen_gray, warp_matrix,
            (img_gen_gray.shape[1], img_gen_gray.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        print("  [ECC] Alignment applied successfully.")
        return aligned
    except cv2.error as e:
        print(f"  [ECC] Alignment failed ({e}) — using unaligned crop.")
        return img_gen_gray


def preprocess_pair(original_path, generated_path, original_bbox=None, generated_bbox=None):
    """
    Loads, crops, resizes, and grayscales both images.

    With bboxes: each image is independently cropped to its own furniture region —
    no perspective alignment needed (both crops already show only their furniture).
    Without bboxes: LoFTR/ORB alignment runs on the full images.
    """
    print(f"  Loading original  : {original_path}")
    img_orig = load_image(original_path)
    print(f"  Loading generated : {generated_path}")
    img_gen  = load_image(generated_path)

    _ensure_dirs()
    base = os.path.splitext(os.path.basename(original_path))[0]

    if original_bbox is not None or generated_bbox is not None:
        if original_bbox is not None:
            x, y, w, h = original_bbox
            print(f"  Cropping original  to bbox: {original_bbox}")
            img_orig = img_orig[y:y+h, x:x+w]
        if generated_bbox is not None:
            x, y, w, h = generated_bbox
            print(f"  Cropping generated to bbox: {generated_bbox}")
            img_gen = img_gen[y:y+h, x:x+w]
        # Resize generated crop to original crop dimensions so both fill the 512×512
        # canvas at the same pixel scale — SSIM compares comparable furniture sizes.
        if original_bbox is not None and generated_bbox is not None:
            oh, ow = img_orig.shape[:2]
            print(f"  Matching generated crop scale → {ow}×{oh}px")
            img_gen = _resize_with_pad(img_gen, ow, oh, pad_val=127)
    else:
        print("  Aligning generated image to original perspective...")
        img_gen = align_images(img_orig, img_gen)
        dw = min(img_orig.shape[1], img_gen.shape[1])
        cv2.imwrite(
            os.path.join(DIFF_MAPS_DIR, f"alignment_debug_{base}.png"),
            np.hstack([cv2.resize(img_orig, (dw, img_orig.shape[0])),
                       cv2.resize(img_gen,  (dw, img_gen.shape[0]))]),
        )

    print(f"  Resizing both to {TARGET_WIDTH}x{TARGET_HEIGHT}px")
    img_orig = resize_image(img_orig)
    img_gen  = resize_image(img_gen)

    if CONVERT_TO_GRAYSCALE:
        print("  Converting to grayscale")
        img_orig = to_grayscale(img_orig)
        img_gen  = to_grayscale(img_gen)

    if original_bbox is not None and generated_bbox is not None:
        print("  Aligning crops with ECC...")
        img_gen = align_crops_ecc(img_orig, img_gen)

    assert img_orig.shape == img_gen.shape, (
        f"Shape mismatch: {img_orig.shape} vs {img_gen.shape}"
    )
    print(f"  Preprocessing done. Shape: {img_orig.shape}")
    return img_orig, img_gen


# ==============================================================
#  SECTION 3 — AUTO BOUNDING BOX DETECTION
# ==============================================================

def _detect_bbox_openai(image):
    """Asks GPT-4o to locate the main wardrobe; returns (x,y,w,h) or None."""
    if not OPENAI_BBOX_ENABLED or not OPENAI_API_KEY:
        return None
    try:
        import base64, json
        from openai import OpenAI
        client       = OpenAI(api_key=OPENAI_API_KEY)
        h_img, w_img = image.shape[:2]
        _, buf       = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64          = base64.b64encode(buf).decode("utf-8")
        resp = client.chat.completions.create(
            model=OPENAI_BBOX_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "This is a product photography image for a furniture company.\n"
                    "Find the MAIN furniture product being sold — it is the largest or most "
                    "prominent piece of furniture in the scene.\n"
                    "Ignore decorative accessories: artwork, vases, bowls, books, plants, "
                    "rugs, cushions, curtains.\n"
                    "Return ONLY JSON:\n"
                    "{\"x\":<left>,\"y\":<top>,\"w\":<width>,\"h\":<height>}\n"
                    f"Image is {w_img}x{h_img}px.\n"
                    "If no furniture found: {\"x\":0,\"y\":0,\"w\":0,\"h\":0}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            max_completion_tokens=60,
        )
        c = json.loads(resp.choices[0].message.content)
        x, y, bw, bh = int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"])
        if bw < 10 or bh < 10:
            print("  [OpenAI bbox] No wardrobe detected.")
            return None
        return (max(0,x), max(0,y), min(w_img-x, bw), min(h_img-y, bh))
    except Exception as e:
        print(f"  [OpenAI bbox] Error ({e}) — falling back to rembg.")
        return None


def _detect_bbox_openai_guided(target_image, reference_crop):
    if not OPENAI_BBOX_ENABLED or not OPENAI_API_KEY:
        return None

    import base64
    import json
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    h_tgt, w_tgt = target_image.shape[:2]

    def _enc(img, quality=85):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    # CALL 1: Describe the reference shape
    try:
        r1 = client.chat.completions.create(
            model=OPENAI_BBOX_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "Look at this furniture item. Describe ONLY its silhouette shape "
                    "in one short sentence. Focus on the outline/profile — is the top "
                    "edge flat, slanted, curved, stepped? Is it taller on left or right? "
                    "Example: 'rectangular box with flat top' or "
                    "'tall cabinet with dramatic diagonal top slanting from tall-left to short-right'. "
                    "One sentence only, no other text."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{_enc(reference_crop)}",
                    "detail": "high"
                }},
            ]}],
            max_completion_tokens=60,
        )
        shape_description = r1.choices[0].message.content.strip()
        print(f"  [guided] Reference shape: {shape_description}")
    except Exception as e:
        print(f"  [guided] Shape description failed ({e})")
        shape_description = "storage cabinet with distinctive silhouette"

    # CALL 2: Find that shape in the target image
    try:
        r2 = client.chat.completions.create(
            model=OPENAI_BBOX_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    f"Find the object matching this shape description: '{shape_description}'\n\n"
                    "IMPORTANT: Match by SILHOUETTE SHAPE ONLY, not by color.\n"
                    "The matching object will have that exact outline profile.\n"
                    "Ignore chairs, lamps, plants, side tables, artwork, windows.\n"
                    "Return ONLY JSON with the bounding box of the shape-matched object:\n"
                    "{\"x\":<left>,\"y\":<top>,\"w\":<width>,\"h\":<height>}\n"
                    f"Image is {w_tgt}x{h_tgt}px.\n"
                    "If not found: {\"x\":0,\"y\":0,\"w\":0,\"h\":0}"
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{_enc(target_image)}",
                    "detail": "high"
                }},
            ]}],
            max_completion_tokens=120,
        )
        c = json.loads(r2.choices[0].message.content)
        x, y, bw, bh = int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"])
        if bw < 10 or bh < 10:
            print("  [guided] Shape not found in target.")
            return None
        print(f"  [guided] Shape match found: x={x}, y={y}, w={bw}, h={bh}")
        return (max(0, x), max(0, y), min(w_tgt-x, bw), min(h_tgt-y, bh))
    except Exception as e:
        print(f"  [guided] Match call failed ({e})")
        return None


def _image_contrast(image_path):
    """Returns LAB L* std-dev as a perceptual contrast measure."""
    l, _, _ = cv2.split(cv2.cvtColor(load_image(image_path), cv2.COLOR_BGR2LAB))
    return float(l.std())


def detect_furniture_bbox(image_path, padding=15, debug=False, label=""):
    """
    Returns (x,y,w,h) of the main furniture object, or None.

    Backend priority: OpenAI Vision → rembg → Sobel+Otsu.
    rembg beats Sobel+Otsu for low-contrast furniture (beige wardrobe on grey wall)
    because it uses appearance-based segmentation instead of gradient contrast.
    The rembg alpha is cached in _rembg_alpha_cache so compute_furniture_mask can
    reuse it without running U2-Net a second time on the same image.
    """
    image        = load_image(image_path)
    h_img, w_img = image.shape[:2]

    def _padded(x, y, bw, bh):
        x  = max(0, x - padding);       y  = max(0, y - padding)
        bw = min(w_img - x, bw + padding * 2)
        bh = min(h_img - y, bh + padding * 2)
        return x, y, bw, bh

    def _debug_save(img, x, y, bw, bh):
        if not debug:
            return
        _ensure_dirs()
        d = img.copy()
        cv2.rectangle(d, (x, y), (x+bw, y+bh), (0,255,0), 3)
        prefix = f"bbox_debug_{label}_" if label else "bbox_debug_"
        cv2.imwrite(os.path.join(DIFF_MAPS_DIR, f"{prefix}{os.path.basename(image_path)}"), d)

    # --- Backend A: OpenAI Vision ---
    if OPENAI_BBOX_ENABLED and OPENAI_API_KEY:
        print("  [OpenAI bbox] Asking GPT-4o to locate the main furniture...")
        result = _detect_bbox_openai(image)
        if result is not None:
            x, y, bw, bh = _padded(*result)
            print(f"  [OpenAI bbox] Detected: x={x}, y={y}, w={bw}, h={bh}  ({bw*bh/(w_img*h_img)*100:.1f}%)")
            _debug_save(image, x, y, bw, bh)
            return (x, y, bw, bh)

    # --- Backend B: rembg (U2-Net) ---
    if REMBG_ENABLED:
        try:
            from rembg import remove as rembg_remove
            from PIL import Image as PILImage

            alpha = np.array(rembg_remove(
                PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ))[:, :, 3]
            _rembg_alpha_cache[image_path] = alpha   # reused by compute_furniture_mask
            mask = np.where(alpha > 127, 255, 0).astype(np.uint8)

            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n_labels < 2:
                print("  [rembg bbox] No foreground objects — using full image.")
                return None

            # Score = area × centrality. In product photography furniture is almost
            # always centred; this prevents a large off-centre lamp from winning.
            cx_img, cy_img = w_img / 2.0, h_img / 2.0
            diag = (w_img**2 + h_img**2) ** 0.5
            best_lbl, best_score = 1, -1.0
            for lbl in range(1, n_labels):
                s    = stats[lbl]
                cx   = s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]  / 2.0
                cy   = s[cv2.CC_STAT_TOP]  + s[cv2.CC_STAT_HEIGHT] / 2.0
                dist = ((cx - cx_img)**2 + (cy - cy_img)**2) ** 0.5
                # Power-4 centrality: a lamp at 28% of the diagonal scores 0.27 vs a
                # cabinet at 9% scoring 0.67 — requiring 2.5× larger area to win instead
                # of the 1.4× the old linear factor-3 formula allowed.
                score = float(s[cv2.CC_STAT_AREA]) * (1.0 - dist / diag) ** 4
                if score > best_score:
                    best_score, best_lbl = score, lbl

            s = stats[best_lbl]
            x, y, bw, bh = _padded(s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_TOP],
                                    s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT])
            print(f"  [rembg bbox] x={x}, y={y}, w={bw}, h={bh}  ({bw*bh/(w_img*h_img)*100:.1f}%)")
            _debug_save(image, x, y, bw, bh)
            return (x, y, bw, bh)
        except Exception as e:
            print(f"  [rembg bbox] Failed ({e}) — falling back to Sobel+Otsu.")

    # --- Backend C: Sobel+Otsu ---
    contrast = _image_contrast(image_path)
    if contrast < AUTO_BBOX_MIN_CONTRAST:
        print(f"  Low contrast (LAB std={contrast:.1f}) — using full image.")
        return None
    print(f"  Contrast OK (LAB std={contrast:.1f}) — detecting furniture...")

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    row_e   = np.convolve(
        np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1),
        np.ones(5)/5, mode="same",
    )
    edge_rows = np.where(row_e > row_e.max() * 0.25)[0]
    if len(edge_rows) < 2:
        print("  Could not detect top/bottom edges — using full image.")
        return None

    top_row, bottom_row = int(edge_rows[0]), int(edge_rows[-1])
    col_s  = np.convolve(gray[top_row:bottom_row].mean(axis=0), np.ones(10)/10, mode="same")
    col_u8 = ((col_s - col_s.min()) / (col_s.max() - col_s.min() + 1e-8) * 255).astype(np.uint8)
    otsu, _ = cv2.threshold(col_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fcols   = np.where(col_u8 > otsu)[0]
    if len(fcols) == 0:
        print("  Could not detect left/right edges — using full image.")
        return None

    left_col, right_col = int(fcols[0]), int(fcols[-1])
    x, y, bw, bh = _padded(left_col, top_row, right_col - left_col, bottom_row - top_row)
    print(f"  Detected bbox: x={x}, y={y}, w={bw}, h={bh}  ({bw*bh/(w_img*h_img)*100:.1f}%)")

    if debug:
        _ensure_dirs()
        d = image.copy()
        cv2.rectangle(d, (x, y), (x+bw, y+bh), (0,255,0), 3)
        cv2.line(d, (0, top_row),    (w_img, top_row),    (0,0,255), 2)
        cv2.line(d, (0, bottom_row), (w_img, bottom_row), (0,0,255), 2)
        cv2.line(d, (left_col,  0),  (left_col,  h_img),  (0,255,255), 2)
        cv2.line(d, (right_col, 0),  (right_col, h_img),  (0,255,255), 2)
        cv2.imwrite(os.path.join(DIFF_MAPS_DIR, f"bbox_debug_{os.path.basename(image_path)}"), d)

    return (x, y, bw, bh)


def detect_generated_bbox(generated_path, original_image, original_bbox, padding=15, debug=False):
    """
    Locates furniture in generated_path using a 3-call GPT approach:
    describe the reference crop, list all objects in the target, then match by description.
    Falls back to detect_furniture_bbox() on any failure.
    """
    gen_img      = load_image(generated_path)
    h_img, w_img = gen_img.shape[:2]

    x0, y0, w0, h0 = original_bbox
    x0, y0 = max(0, x0), max(0, y0)
    w0, h0 = min(w0, original_image.shape[1]-x0), min(h0, original_image.shape[0]-y0)
    reference_crop = original_image[y0:y0+h0, x0:x0+w0]

    if not (OPENAI_BBOX_ENABLED and OPENAI_API_KEY):
        print("  [guided] OpenAI not configured — falling back to standard detection.")
        return detect_furniture_bbox(generated_path, padding=padding, debug=debug, label="generated")

    import base64
    import json
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    def _enc(img, quality=85):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode("utf-8")

    # CALL 1 — describe the reference crop
    try:
        r1 = client.chat.completions.create(
            model=OPENAI_BBOX_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "Describe this furniture piece in one sentence covering:\n"
                    "1. Its TYPE (e.g. tall wardrobe, long low sideboard, TV cabinet, "
                    "chest of drawers, bookshelf)\n"
                    "2. Its SHAPE (e.g. very wide and low, tall and narrow, square)\n"
                    "3. Its COLOR (e.g. white, oak wood, dark grey)\n"
                    "Reply with ONE sentence only."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{_enc(reference_crop)}",
                    "detail": "high",
                }},
            ]}],
            max_completion_tokens=80,
        )
        description = r1.choices[0].message.content.strip()
        print(f"  [guided] Reference description: {description}")
    except Exception as e:
        print(f"  [guided] Description call failed ({e}) — falling back.")
        return detect_furniture_bbox(generated_path, padding=padding, debug=debug, label="generated")

    # CALL 2 — list ALL objects in target image
    try:
        r2 = client.chat.completions.create(
            model=OPENAI_BBOX_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "List every distinct object in this image as a JSON object with an 'objects' array.\n"
                    "Each item: {\"label\":\"name\",\"x\":n,\"y\":n,\"w\":n,\"h\":n}\n"
                    "Include ALL furniture, art, plants, lamps, decorations — everything.\n"
                    f"Image is {w_img}x{h_img}px.\n"
                    "Return ONLY: {\"objects\": [...]}"
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{_enc(gen_img)}",
                    "detail": "high",
                }},
            ]}],
            max_completion_tokens=600,
        )
        objects = json.loads(r2.choices[0].message.content).get("objects", [])
        if not objects:
            print("  [guided] No objects detected — falling back.")
            return detect_furniture_bbox(generated_path, padding=padding, debug=debug, label="generated")
        print(f"  [guided] Detected {len(objects)} objects in scene.")
    except Exception as e:
        print(f"  [guided] Object listing call failed ({e}) — falling back.")
        return detect_furniture_bbox(generated_path, padding=padding, debug=debug, label="generated")

    # CALL 3 — match by description
    try:
        numbered = "\n".join(f"{i+1}. {o['label']}" for i, o in enumerate(objects))
        r3 = client.chat.completions.create(
            model=OPENAI_BBOX_MODEL,
            messages=[{"role": "user", "content": (
                f"A furniture piece is described as: '{description}'\n\n"
                f"Numbered objects in the scene:\n{numbered}\n\n"
                "Which number is this furniture piece?\n"
                "Consider TYPE and SHAPE as most important — ignore color differences.\n"
                "A 'long low sideboard' cannot match a 'picture frame' or 'plant'.\n"
                "Reply with ONLY the number, nothing else."
            )}],
            max_completion_tokens=5,
        )
        idx = int(r3.choices[0].message.content.strip()) - 1
        if not (0 <= idx < len(objects)):
            print(f"  [guided] Match index {idx+1} out of range — falling back.")
            return detect_furniture_bbox(generated_path, padding=padding, debug=debug, label="generated")
        matched = objects[idx]
        print(f"  [guided] Matched object #{idx+1}: {matched['label']}")
    except Exception as e:
        print(f"  [guided] Match call failed ({e}) — falling back.")
        return detect_furniture_bbox(generated_path, padding=padding, debug=debug, label="generated")

    x  = max(0, int(matched["x"]) - padding)
    y  = max(0, int(matched["y"]) - padding)
    bw = min(w_img - x, int(matched["w"]) + padding * 2)
    bh = min(h_img - y, int(matched["h"]) + padding * 2)
    print(f"  [guided] Found: x={x}, y={y}, w={bw}, h={bh}  ({bw*bh/(w_img*h_img)*100:.1f}%)")

    if debug:
        _ensure_dirs()
        d = gen_img.copy()
        cv2.rectangle(d, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(DIFF_MAPS_DIR,
            f"bbox_debug_generated_{os.path.basename(generated_path)}"), d)

    return (x, y, bw, bh)


# ==============================================================
#  SECTION 4 — FURNITURE MASKING
# ==============================================================

def compute_furniture_mask(image, bbox, target_size=(TARGET_WIDTH, TARGET_HEIGHT), cached_alpha=None):
    """
    Returns a binary (0/255) foreground mask covering only the furniture pixels.

    rembg (U2-Net) handles low-contrast scenes where GrabCut fails (white cabinet on white
    wall). Pass cached_alpha to reuse the alpha already computed by detect_furniture_bbox()
    and avoid running U2-Net twice on the same image.
    """
    img_h, img_w = image.shape[:2]
    x, y, bw, bh = bbox

    def _crop_and_resize(full_mask):
        cx  = max(0, x);         cy  = max(0, y)
        cbw = min(bw, img_w-cx); cbh = min(bh, img_h-cy)
        resized = _resize_with_pad(full_mask[cy:cy+cbh, cx:cx+cbw],
                                   target_size[0], target_size[1],
                                   pad_val=0, interpolation=cv2.INTER_NEAREST)
        _, out = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        return out

    if REMBG_ENABLED:
        try:
            if cached_alpha is not None:
                alpha = cached_alpha
                print("  [mask] Using cached rembg alpha (U2-Net skipped).")
            else:
                from rembg import remove as rembg_remove
                from PIL import Image as PILImage
                alpha = np.array(rembg_remove(
                    PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ))[:, :, 3]

            furniture_mask = np.where(alpha > 127, 255, 0).astype(np.uint8)
            coverage = int(furniture_mask.sum()) // 255 / (img_h * img_w)
            if not (0.05 <= coverage <= 0.95):
                print(f"  [mask] rembg coverage {coverage:.1%} unreliable — using full image.")
                return None
            print(f"  [mask] rembg mask: {coverage:.1%} of image.")
            return _crop_and_resize(furniture_mask)

        except ImportError:
            print("  [mask] rembg not installed — falling back to GrabCut.")
        except Exception as e:
            print(f"  [mask] rembg error ({e}) — falling back to GrabCut.")

    # GrabCut fallback
    cx  = max(0, x);         cy  = max(0, y)
    cbw = min(bw, img_w-cx); cbh = min(bh, img_h-cy)
    gc_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    try:
        cv2.grabCut(image, gc_mask, (cx, cy, cbw, cbh),
                    np.zeros((1,65), np.float64), np.zeros((1,65), np.float64),
                    iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"  [mask] GrabCut error ({e}) — using full image.")
        return None

    furniture_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    coverage = int(furniture_mask.sum()) // 255 / (img_h * img_w)
    if not (0.10 <= coverage <= 0.90):
        print(f"  [mask] GrabCut coverage {coverage:.1%} unreliable — using full image.")
        return None
    print(f"  [mask] GrabCut mask: {coverage:.1%} of image.")
    return _crop_and_resize(furniture_mask)


# ==============================================================
#  SECTION 5 — COMPARISON
# ==============================================================

def _ensure_dirs():
    os.makedirs(DIFF_MAPS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def _diff_to_visual(diff):
    """Converts SSIM diff map (1=identical) to JET heatmap (red=problem)."""
    return cv2.applyColorMap(cv2.bitwise_not((diff * 255).astype("uint8")), cv2.COLORMAP_JET)


def compute_ssim(img1, img2, mask=None, mask2=None):
    """
    Computes SSIM on Canny edge maps, masked to the furniture region.

    Two-layer background elimination:
      1. Canny edges: flat background produces no edges → zero SSIM contribution.
      2. Furniture mask: zeros remaining background edges (skirting, lamps, wall).
    Same wardrobe, different rooms: no mask → SSIM≈0.47; edge+mask → SSIM≈0.81.
    """
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both images must be grayscale (2D).")
    if img1.shape != img2.shape:
        raise ValueError(f"Shapes must match: {img1.shape} vs {img2.shape}.")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges1 = cv2.dilate(cv2.Canny(cv2.GaussianBlur(img1, (3,3), 0), 30, 100), kernel)
    edges2 = cv2.dilate(cv2.Canny(cv2.GaussianBlur(img2, (3,3), 0), 30, 100), kernel)

    def _apply(edges, m):
        if m is None:
            return edges
        if m.shape != edges.shape:
            m = _resize_with_pad(m, edges.shape[1], edges.shape[0],
                                 pad_val=0, interpolation=cv2.INTER_NEAREST)
        return cv2.bitwise_and(edges, edges, mask=m)

    edges1 = _apply(edges1, mask)
    edges2 = _apply(edges2, mask2 if mask2 is not None else mask)

    win_size = SSIM_WIN_SIZE
    min_dim  = min(edges1.shape)
    if win_size >= min_dim:
        win_size = min_dim - 1 if (min_dim - 1) % 2 != 0 else min_dim - 2
        print(f"  Note: win_size reduced to {win_size}.")

    score, diff = ssim(edges1, edges2, full=True, win_size=win_size, data_range=255)
    return score, diff


def compute_fsim(img1, img2):
    """Returns FSIM score, or None if disabled / library missing."""
    if not FSIM_ENABLED:
        return None
    try:
        from image_similarity_measures.quality_metrics import fsim as _fsim
        return float(_fsim(img1, img2))
    except ImportError:
        print("  [FSIM] library not installed (pip install image-similarity-measures).")
        return None


def compute_lpips(img1_bgr, img2_bgr):
    """
    Returns LPIPS perceptual distance (0=identical, 1=completely different).
    Minor angle/lighting shifts score low; structural defects score high.
    Returns None if disabled or lpips/torch not installed.
    """
    if not LPIPS_ENABLED:
        return None
    try:
        import torch
        import lpips as lpips_lib
    except ImportError:
        print("  [LPIPS] lpips or torch not installed.")
        return None

    global _lpips_loss_fn
    if _lpips_loss_fn is None:
        _lpips_loss_fn = lpips_lib.LPIPS(net=LPIPS_NET, verbose=False)
        _lpips_loss_fn.eval()

    def _to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)

    size = (TARGET_WIDTH, TARGET_HEIGHT)
    with torch.no_grad():
        return float(_lpips_loss_fn(
            _to_tensor(cv2.resize(img1_bgr, size)),
            _to_tensor(cv2.resize(img2_bgr, size)),
        ).item())


_lpips_loss_fn   = None
_rembg_alpha_cache: dict = {}   # image_path → alpha; avoids running U2-Net twice per image


def compute_hash_similarity(img1_bgr, img2_bgr):
    """
    Returns (similarity, hamming_distance) using dHash.
    dHash encodes gradient structure as a HASH_SIZE² fingerprint — robust to minor
    scale/brightness shifts but sensitive to structural changes (missing shelf, different silhouette).
    Returns (None, None) if disabled or imagehash not installed.
    """
    if not HASH_ENABLED:
        return None, None
    try:
        import imagehash
        from PIL import Image as PILImage
    except ImportError:
        print("  [Hash] imagehash or Pillow not installed.")
        return None, None

    def _pil(bgr):
        return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    h1, h2  = imagehash.dhash(_pil(img1_bgr), HASH_SIZE), imagehash.dhash(_pil(img2_bgr), HASH_SIZE)
    hamming = h1 - h2
    return 1.0 - hamming / (HASH_SIZE * HASH_SIZE), hamming


def save_diff_map(diff, image_name):
    _ensure_dirs()
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(DIFF_MAPS_DIR, f"{image_name}_diff_{ts}.png")
    cv2.imwrite(path, _diff_to_visual(diff))
    return path


def save_comparison_image(img_original, img_generated, diff, image_name):
    """Saves a 3-panel [ ORIGINAL | GENERATED | DIFF MAP ] image."""
    _ensure_dirs()

    def _label(img, text):
        banner = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, text, (10,28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,255), 2, cv2.LINE_AA)
        return np.vstack([banner, img])

    comparison = np.hstack([
        _label(cv2.cvtColor(img_original,  cv2.COLOR_GRAY2BGR), "ORIGINAL"),
        _label(cv2.cvtColor(img_generated, cv2.COLOR_GRAY2BGR), "GENERATED"),
        _label(_diff_to_visual(diff),                           "DIFF MAP (red=problem)"),
    ])
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(DIFF_MAPS_DIR, f"{image_name}_comparison_{ts}.png")
    cv2.imwrite(path, comparison)
    return path


def find_best_angle_ssim(img_orig, img_gen, mask=None, mask2=None):
    """
    Tries small rotations of img_gen (-15° to +15° in 1° steps) and returns
    the rotation angle that maximizes SSIM score, plus that score and diff.
    Uses a coarse-then-fine search: first every 3°, then ±3° around the best.
    """
    h, w = img_gen.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def _rotate(img, angle):
        if angle == 0:
            return img
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # Coarse pass: every 3° from -15 to +15
    best_angle = 0
    best_score = -1.0
    best_diff  = None

    for angle in range(-15, 16, 3):
        rotated = _rotate(img_gen, angle)
        score, diff = compute_ssim(img_orig, rotated, mask=mask, mask2=mask2)
        if score > best_score:
            best_score = score
            best_angle = angle
            best_diff  = diff

    # Fine pass: every 1° around best coarse angle
    for angle in range(best_angle - 3, best_angle + 4, 1):
        if angle % 3 == 0:
            continue  # already tested
        rotated = _rotate(img_gen, angle)
        score, diff = compute_ssim(img_orig, rotated, mask=mask, mask2=mask2)
        if score > best_score:
            best_score = score
            best_angle = angle
            best_diff  = diff

    print(f"  [angle] Best rotation: {best_angle}°  →  SSIM {best_score:.4f}")
    return best_angle, best_score, best_diff


def compare_images(img_original, img_generated, image_name="image",
                   mask=None, mask_generated=None,
                   lpips_score=None, hash_similarity=None, hash_distance=None):
    """
    Runs the full scoring stack and returns a result dict.

    Verdict: PASS iff ssim_passed AND (lpips_passed or None) AND (hash_passed or None).
    FSIM gates the verdict when enabled (treated same as secondary metrics).
    lpips_score and hash_* are pre-computed on BGR images in run_qa() and passed in
    because compare_images only receives grayscale.
    """
    print(f"\n  Comparing: {image_name}")

    ssim_score, diff = compute_ssim(img_original, img_generated, mask=mask, mask2=mask_generated)
    ssim_passed = ssim_score >= SSIM_THRESHOLD
    print(f"  SSIM  : {ssim_score:.4f}  (threshold {SSIM_THRESHOLD})  →  {'✅ PASS' if ssim_passed else '❌ FAIL'}")

    # Try angle correction if SSIM failed
    best_angle = 0
    if not ssim_passed:
        print("  SSIM failed — trying angle correction...")
        best_angle, corrected_score, corrected_diff = find_best_angle_ssim(
            img_original, img_generated, mask=mask, mask2=mask_generated
        )
        if best_angle != 0 and corrected_score > ssim_score:
            print(f"  [angle] Improvement: {ssim_score:.4f} → {corrected_score:.4f} at {best_angle}°")
            ssim_score  = corrected_score
            diff        = corrected_diff
            ssim_passed = ssim_score >= SSIM_THRESHOLD

    fsim_score  = compute_fsim(img_original, img_generated)
    fsim_passed = None
    if fsim_score is not None:
        fsim_passed = fsim_score >= FSIM_THRESHOLD
        print(f"  FSIM  : {fsim_score:.4f}  (threshold {FSIM_THRESHOLD})  →  {'✅ PASS' if fsim_passed else '❌ FAIL'}")

    lpips_passed = None
    if lpips_score is not None:
        lpips_passed = lpips_score <= LPIPS_THRESHOLD
        print(f"  LPIPS : {lpips_score:.4f}  (threshold ≤{LPIPS_THRESHOLD})  →  {'✅ PASS' if lpips_passed else '❌ FAIL'}")

    hash_passed = None
    if hash_similarity is not None and hash_distance is not None:
        hash_passed = hash_distance <= HASH_THRESHOLD
        print(f"  Hash  : {hash_similarity:.4f}  (hamming={hash_distance}, threshold ≤{HASH_THRESHOLD})  →  {'✅ PASS' if hash_passed else '❌ FAIL'}")

    warnings = []
    if FSIM_ENABLED  and fsim_score      is None: warnings.append("FSIM skipped — install image-similarity-measures")
    if LPIPS_ENABLED and lpips_score     is None: warnings.append("LPIPS skipped — install lpips torch")
    if HASH_ENABLED  and hash_similarity is None: warnings.append("Hash skipped — install imagehash")
    if warnings:
        print("  ⚠️  Warnings:")
        for w in warnings:
            print(f"       • {w}")

    all_passed = (
        ssim_passed
        and (fsim_passed  is None or fsim_passed)
        and (lpips_passed is None or lpips_passed)
        and (hash_passed  is None or hash_passed)
    )
    verdict = "PASS" if all_passed else "FAIL"

    comparison_path = save_comparison_image(img_original, img_generated, diff, image_name)
    diff_map_path   = save_diff_map(diff, image_name) if (not all_passed or SAVE_DIFF_MAP) else None

    if not all_passed and diff_map_path:
        print(f"  Diff map   → {diff_map_path}")
    print(f"  Comparison → {comparison_path}")

    if all_passed:
        message = f"PASS — Structure consistent. SSIM: {ssim_score:.4f}"
    else:
        reasons = []
        if not ssim_passed:       reasons.append(f"SSIM {ssim_score:.4f} < {SSIM_THRESHOLD}")
        if fsim_passed  is False: reasons.append(f"FSIM {fsim_score:.4f} < {FSIM_THRESHOLD}")
        if lpips_passed is False: reasons.append(f"LPIPS {lpips_score:.4f} > {LPIPS_THRESHOLD}")
        if hash_passed  is False: reasons.append(f"Hash hamming={hash_distance} > {HASH_THRESHOLD}")
        message = "FAIL — Structural deformation detected! " + " | ".join(reasons)

    return {
        "image_name"     : image_name,
        "ssim_score"     : round(ssim_score, 4),
        "ssim_passed"    : ssim_passed,
        "fsim_score"     : round(fsim_score, 4)      if fsim_score      is not None else None,
        "fsim_passed"    : fsim_passed,
        "lpips_score"    : round(lpips_score, 4)     if lpips_score     is not None else None,
        "lpips_passed"   : lpips_passed,
        "hash_similarity": round(hash_similarity, 4) if hash_similarity is not None else None,
        "hash_distance"  : hash_distance,
        "hash_passed"    : hash_passed,
        "best_angle"     : best_angle,
        "passed"         : all_passed,
        "verdict"        : verdict,
        "message"        : message,
        "warnings"       : warnings,
        "diff_map_path"  : diff_map_path,
        "comparison_path": comparison_path,
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ==============================================================
#  SECTION 5 — LOGGING
# ==============================================================

def log_result(result):
    _ensure_dirs()
    line = (
        f"[{result['timestamp']}] "
        f"{result['verdict']:4}  |  "
        f"{result['image_name']:30}  |  "
        f"SSIM: {result['ssim_score']:.4f}"
    )
    if result["fsim_score"]    is not None: line += f"  FSIM: {result['fsim_score']:.4f}"
    if result["lpips_score"]   is not None: line += f"  LPIPS: {result['lpips_score']:.4f}"
    if result["hash_distance"] is not None: line += f"  Hash: {result['hash_distance']}bits"
    if result.get("best_angle"):            line += f"  Angle: {result['best_angle']}°"
    if not result["passed"] and result["diff_map_path"]:
        line += f"  |  diff → {result['diff_map_path']}"
    print(f"\n  {line}")
    if LOG_TO_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ==============================================================
#  SECTION 6 — RUNNER
# ==============================================================

def _save_bbox_comparison_debug(original_path, generated_path, original_bbox, generated_bbox, image_name):
    """Saves a 3-panel debug image: full original+bbox | full generated+bbox | matched crops."""
    _ensure_dirs()
    orig_img = load_image(original_path)
    gen_img  = load_image(generated_path)

    orig_m, gen_m = orig_img.copy(), gen_img.copy()
    if original_bbox  is not None:
        x, y, w, h = original_bbox;  cv2.rectangle(orig_m, (x,y), (x+w,y+h), (0,255,0), 4)
    if generated_bbox is not None:
        x, y, w, h = generated_bbox; cv2.rectangle(gen_m,  (x,y), (x+w,y+h), (0,255,0), 4)

    disp_orig = _resize_with_pad(orig_m, TARGET_WIDTH, TARGET_HEIGHT, pad_val=127)
    disp_gen  = _resize_with_pad(gen_m,  TARGET_WIDTH, TARGET_HEIGHT, pad_val=127)

    if original_bbox is not None and generated_bbox is not None:
        ox, oy, ow, oh = original_bbox
        gx, gy, gw, gh = generated_bbox
        crop_orig = orig_img[oy:oy+oh, ox:ox+ow]
        # Resize generated crop to original crop dims so both appear at the same scale,
        # honestly reflecting what SSIM actually compares.
        crop_gen  = _resize_with_pad(gen_img[gy:gy+gh, gx:gx+gw], ow, oh, pad_val=127)
        crop_panel = np.hstack([
            _resize_with_pad(crop_orig, TARGET_WIDTH, TARGET_HEIGHT, pad_val=127),
            _resize_with_pad(crop_gen,  TARGET_WIDTH, TARGET_HEIGHT, pad_val=127),
        ])
    else:
        crop_panel = np.full((TARGET_HEIGHT, TARGET_WIDTH*2, 3), 80, dtype=np.uint8)

    def _lbl(img, text, scale=0.75, color=(255,255,255)):
        b = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(b, text, (8,26), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return np.vstack([b, img])

    crop_label = np.zeros((36, crop_panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(crop_label, "CROPS (what SSIM compares — same scale)",
                (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2, cv2.LINE_AA)

    combined = np.hstack([
        np.hstack([_lbl(disp_orig, "ORIGINAL"), _lbl(disp_gen, "GENERATED")]),
        np.vstack([crop_label, crop_panel]),
    ])
    out = os.path.join(DIFF_MAPS_DIR, f"bbox_comparison_{image_name}.png")
    cv2.imwrite(out, combined)
    print(f"  Bbox comparison debug saved: {out}")


def run_qa(original_path, generated_path, auto_bbox=False, debug_bbox=False):
    """
    Full pipeline: detect bbox → segment mask → compute LPIPS+dHash → preprocess → SSIM → log.

    LPIPS and dHash are computed on bbox-cropped, masked BGR images so the room background
    (different wall colours, lamps, plants) does not inflate the perceptual distance.
    rembg alpha is cached from bbox detection so U2-Net runs at most once per image.
    """
    image_name = os.path.splitext(os.path.basename(original_path))[0]
    print("\n" + "="*55)
    print(f"  QA Check: {image_name}")
    print("="*55)

    original_bbox = generated_bbox = original_mask = generated_mask = None

    if auto_bbox:
        backend = ("OpenAI" if (OPENAI_BBOX_ENABLED and OPENAI_API_KEY)
                   else "rembg" if REMBG_ENABLED else "Sobel+Otsu")

        print(f"  [{backend}] Detecting furniture bbox in original...")
        original_bbox = detect_furniture_bbox(original_path, debug=debug_bbox, label="original")

        if original_bbox is not None and OPENAI_BBOX_ENABLED and OPENAI_API_KEY:
            generated_bbox = detect_generated_bbox(
                generated_path, load_image(original_path), original_bbox, debug=debug_bbox,
            )
        else:
            print(f"  [{backend}] Detecting furniture bbox in generated (standard)...")
            generated_bbox = detect_furniture_bbox(generated_path, debug=debug_bbox, label="generated")

        if original_bbox is None and generated_bbox is None:
            print("  Both bbox detections failed — using full images.")

        if debug_bbox:
            _save_bbox_comparison_debug(original_path, generated_path,
                                        original_bbox, generated_bbox, image_name)

        seg = "rembg" if REMBG_ENABLED else "GrabCut"
        if original_bbox is not None:
            print(f"  [{seg}] Segmenting furniture in original...")
            original_mask = compute_furniture_mask(
                load_image(original_path), original_bbox,
                cached_alpha=_rembg_alpha_cache.get(original_path),
            )
        if generated_bbox is not None:
            print(f"  [{seg}] Segmenting furniture in generated...")
            generated_mask = compute_furniture_mask(
                load_image(generated_path), generated_bbox,
                cached_alpha=_rembg_alpha_cache.get(generated_path),
            )

    lpips_score = hash_similarity = hash_distance = None

    if LPIPS_ENABLED or HASH_ENABLED:
        orig_bgr = load_image(original_path)
        gen_bgr  = load_image(generated_path)

        if original_bbox  is not None:
            x, y, w, h = original_bbox;  orig_bgr = orig_bgr[y:y+h, x:x+w]
        if generated_bbox is not None:
            x, y, w, h = generated_bbox; gen_bgr  = gen_bgr[y:y+h, x:x+w]

        def _mask_bgr(bgr, m):
            if m is None:
                return bgr
            bh, bw = bgr.shape[:2]
            mf = _resize_with_pad(m, bw, bh, pad_val=0, interpolation=cv2.INTER_NEAREST)
            return cv2.bitwise_and(bgr, cv2.merge([mf]*3))

        orig_bgr = _resize_with_pad(_mask_bgr(orig_bgr, original_mask),  TARGET_WIDTH, TARGET_HEIGHT, pad_val=0)
        gen_bgr  = _resize_with_pad(_mask_bgr(gen_bgr,  generated_mask), TARGET_WIDTH, TARGET_HEIGHT, pad_val=0)

        if LPIPS_ENABLED:
            print("  Computing LPIPS...")
            lpips_score = compute_lpips(orig_bgr, gen_bgr)
            if lpips_score is not None:
                print(f"  [LPIPS] distance = {lpips_score:.4f}")

        if HASH_ENABLED:
            print("  Computing dHash...")
            hash_similarity, hash_distance = compute_hash_similarity(orig_bgr, gen_bgr)
            if hash_distance is not None:
                print(f"  [Hash] hamming = {hash_distance}/{HASH_SIZE**2} bits")

    img_orig, img_gen = preprocess_pair(
        original_path, generated_path,
        original_bbox=original_bbox, generated_bbox=generated_bbox,
    )
    result = compare_images(
        img_orig, img_gen, image_name,
        mask=original_mask, mask_generated=generated_mask,
        lpips_score=lpips_score, hash_similarity=hash_similarity, hash_distance=hash_distance,
    )
    log_result(result)
    print(f"\n  ► {result['message']}")
    return result


# ==============================================================
#  SECTION 7 — IMAGE PAIRS
#  Add pairs here. Run: python qa_tool.py
#    auto_bbox  True  = auto-detect furniture region, ignore background
#    debug_bbox True  = save debug image showing detected box
# ==============================================================

IMAGE_PAIRS = [
    {"name": "cabinet_01", "original": "images/originals/5.png",  "generated": "images/generated/5.png",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_02", "original": "images/originals/1.jpg",  "generated": "images/generated/1.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_03", "original": "images/originals/2.jpg",  "generated": "images/generated/2.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_04", "original": "images/originals/3.jpg",  "generated": "images/generated/3.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_05", "original": "images/originals/4.jpg",  "generated": "images/generated/4.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_06", "original": "images/originals/6.jpg",  "generated": "images/generated/6.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_07", "original": "images/originals/7.jpg",  "generated": "images/generated/7.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_08", "original": "images/originals/8.jpg",  "generated": "images/generated/8.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_09", "original": "images/originals/9.jpg",  "generated": "images/generated/9.jpg",  "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_10", "original": "images/originals/10.png", "generated": "images/generated/10.png", "bbox": None, "auto_bbox": True, "debug_bbox": True},
    {"name": "cabinet_11", "original": "images/originals/11.png", "generated": "images/generated/11.png", "bbox": None, "auto_bbox": True, "debug_bbox": True},
]


def run_batch(pairs):
    """Runs QA on every pair and prints a summary table + per-metric pass rates."""
    results = []
    for pair in pairs:
        result = run_qa(pair["original"], pair["generated"],
                        auto_bbox=pair.get("auto_bbox", False),
                        debug_bbox=pair.get("debug_bbox", False))
        result["image_name"] = pair["name"]
        results.append(result)

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    total  = len(results)

    print("\n" + "="*55)
    print("  BATCH SUMMARY")
    print("="*55)
    print(f"  Total   : {total}")
    print(f"  ✅ PASS : {len(passed)}")
    print(f"  ❌ FAIL : {len(failed)}")
    print("-"*55)

    for r in results:
        icon  = "✅" if r["passed"] else "❌"
        extra = ""
        if r["fsim_score"]    is not None: extra += f"  FSIM: {r['fsim_score']:.4f}"
        if r["lpips_score"]   is not None: extra += f"  LPIPS: {r['lpips_score']:.4f}"
        if r["hash_distance"] is not None: extra += f"  Hash: {r['hash_distance']}b"
        print(f"  {icon}  {r['verdict']:4}  |  {r['image_name']:30}  |  SSIM: {r['ssim_score']:.4f}{extra}")

    if failed:
        print("\n  Failed images — check diff maps:")
        for r in failed:
            print(f"    → {r['image_name']} : {r['diff_map_path']}")

    print("\n  METRIC BREAKDOWN")
    print("-"*55)
    print(f"  SSIM  : {sum(1 for r in results if r['ssim_passed'])}/{total} passed")
    lp = [r for r in results if r["lpips_passed"] is not None]
    if lp:
        print(f"  LPIPS : {sum(1 for r in lp if r['lpips_passed'])}/{len(lp)} passed")
    hh = [r for r in results if r["hash_passed"] is not None]
    if hh:
        print(f"  Hash  : {sum(1 for r in hh if r['hash_passed'])}/{len(hh)} passed")
    print("="*55)

    return results


if __name__ == "__main__":
    run_batch(IMAGE_PAIRS)
