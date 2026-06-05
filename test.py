import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Section 1: Image Alignment
# ---------------------------------------------------------------------------

def align_images(img_original, img_generated):
    """
    Align img_generated to match the perspective of img_original using ORB
    feature matching and homography estimation.

    This corrects for differences in camera angle, zoom, or perspective
    between the two images before structural comparison. For example, if a
    wardrobe photo was taken from slightly to the left in one image and
    straight-on in another, this function warps the generated image so it
    matches the original's viewpoint before SSIM is computed.

    Pipeline:
        grayscale → ORB detection → BFMatcher → ratio test →
        homography (RANSAC) → warpPerspective → aligned image

    If alignment fails at any step (too few matches, degenerate homography),
    the function logs a warning and returns img_generated resized to match
    img_original's dimensions, so the caller never receives a crash.

    Parameters
    ----------
    img_original : np.ndarray
        The reference image loaded by cv2.imread (BGR colour, uint8).
    img_generated : np.ndarray
        The image to be warped toward the original's viewpoint (BGR, uint8).

    Returns
    -------
    np.ndarray
        The perspective-corrected version of img_generated with the same
        height and width as img_original. Returns img_generated resized to
        those dimensions if a reliable alignment cannot be computed.
    """

    # Minimum number of high-quality feature matches needed before we attempt
    # to compute a homography. A homography requires exactly 4 point pairs,
    # but 10+ matches give RANSAC enough data to reject bad outliers reliably.
    MIN_GOOD_MATCHES = 10

    # Lowe's ratio test threshold. A match is kept only when the distance to
    # the best match is less than 75 % of the distance to the second-best.
    # This discards ambiguous matches that are likely to be false positives.
    RATIO_THRESHOLD = 0.75

    # Target canvas size: the aligned image must match the original exactly
    # so that SSIM can compare them pixel-for-pixel afterward.
    original_height, original_width = img_original.shape[:2]

    # ---------------------------------------------------------------------- #
    # Step 1: Convert both images to grayscale for feature detection          #
    # ---------------------------------------------------------------------- #
    # ORB detects corners and edges based on intensity gradients, not colour.
    # Converting to grayscale first removes colour differences between the two
    # images so the detector focuses purely on structural features.
    gray_original  = cv2.cvtColor(img_original,  cv2.COLOR_BGR2GRAY)
    gray_generated = cv2.cvtColor(img_generated, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------------------- #
    # Step 2: Detect ORB keypoints and compute binary descriptors             #
    # ---------------------------------------------------------------------- #
    # ORB (Oriented FAST and Rotated BRIEF) is a patent-free feature detector.
    # It finds distinctive points such as corners and edges, then encodes a
    # compact 256-bit binary descriptor for each one.
    # nfeatures=2000 gives enough candidates even for texturally sparse images
    # (e.g. plain white wardrobes with only handles as distinguishing features).
    orb_detector = cv2.ORB_create(nfeatures=2000)

    keypoints_original,  descriptors_original  = orb_detector.detectAndCompute(gray_original,  None)
    keypoints_generated, descriptors_generated = orb_detector.detectAndCompute(gray_generated, None)

    # Guard: if either image yields no descriptors (e.g. completely blank image),
    # alignment is impossible so we return the resized generated image as-is.
    if descriptors_original is None or descriptors_generated is None:
        print("[align_images] WARNING: No descriptors found in one or both images. "
              "Skipping alignment — falling back to resized generated image.")
        fallback_image = cv2.resize(img_generated, (original_width, original_height))
        return fallback_image

    # ---------------------------------------------------------------------- #
    # Step 3: Match descriptors with Brute-Force Matcher + Hamming distance   #
    # ---------------------------------------------------------------------- #
    # BFMatcher exhaustively compares every descriptor in img_generated against
    # every descriptor in img_original to find the closest pair.
    # NORM_HAMMING is correct for ORB's binary descriptors — it counts the
    # number of differing bits. (L2 norm is used for float descriptors like SIFT.)
    # crossCheck=False because we apply Lowe's ratio test in the next step,
    # which needs the two nearest neighbours, not just one.
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # knnMatch returns a list of (best_match, second_best_match) pairs — one
    # pair per descriptor in img_generated.
    raw_match_pairs = bf_matcher.knnMatch(descriptors_generated, descriptors_original, k=2)

    # ---------------------------------------------------------------------- #
    # Step 4: Apply Lowe's ratio test to keep only reliable matches           #
    # ---------------------------------------------------------------------- #
    # For each candidate match, we keep it only when the nearest neighbour is
    # substantially closer than the second-nearest. A ratio above 0.75 means
    # the match is ambiguous and likely a false correspondence.
    good_matches = []
    for match_pair in raw_match_pairs:
        # knnMatch can return fewer than k=2 neighbours when the descriptor
        # set is very small; guard against an IndexError when unpacking.
        if len(match_pair) < 2:
            continue
        best_match, second_best_match = match_pair
        if best_match.distance < RATIO_THRESHOLD * second_best_match.distance:
            good_matches.append(best_match)

    print(f"[align_images] Found {len(good_matches)} good feature matches "
          f"(minimum required: {MIN_GOOD_MATCHES}).")

    # Guard: too few matches means the two images share too little visual
    # structure to compute a meaningful homography (e.g. completely different
    # furniture models). Return the fallback rather than a garbage warp.
    if len(good_matches) < MIN_GOOD_MATCHES:
        print("[align_images] WARNING: Too few good matches to compute a reliable "
              "homography. Skipping alignment — falling back to resized generated image.")
        fallback_image = cv2.resize(img_generated, (original_width, original_height))
        return fallback_image

    # ---------------------------------------------------------------------- #
    # Step 5: Extract matched keypoint coordinates                            #
    # ---------------------------------------------------------------------- #
    # findHomography needs two arrays of (x, y) point pairs:
    #   src_points — locations in img_generated (the image we want to warp)
    #   dst_points — corresponding locations in img_original (the target plane)
    src_points = np.float32(
        [keypoints_generated[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [keypoints_original[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    # ---------------------------------------------------------------------- #
    # Step 6: Compute the homography matrix with RANSAC                       #
    # ---------------------------------------------------------------------- #
    # findHomography fits a 3×3 perspective transformation matrix that maps
    # src_points onto dst_points.  RANSAC automatically discards outlier pairs
    # (bad matches that survived the ratio test), making the result robust even
    # when 30–40 % of matches are incorrect.
    # ransacReprojThreshold=5.0 treats a match as an inlier if the reprojection
    # error is within 5 pixels — a good balance between strictness and tolerance.
    homography_matrix, inlier_mask = cv2.findHomography(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    # Guard: findHomography returns None when it cannot find any solution,
    # which can happen with degenerate point configurations (e.g. all points
    # on a single line). Fall back gracefully rather than crashing.
    if homography_matrix is None:
        print("[align_images] WARNING: Homography computation returned None. "
              "Skipping alignment — falling back to resized generated image.")
        fallback_image = cv2.resize(img_generated, (original_width, original_height))
        return fallback_image

    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    print(f"[align_images] Homography computed successfully with {inlier_count} inliers.")

    # ---------------------------------------------------------------------- #
    # Step 7: Warp the generated image onto the original's perspective plane  #
    # ---------------------------------------------------------------------- #
    # warpPerspective applies the homography to every pixel of img_generated,
    # transforming its perspective so it matches img_original's viewpoint.
    # The output canvas is sized to exactly match img_original so SSIM can
    # compare them without any further resizing.
    aligned_generated = cv2.warpPerspective(
        img_generated,
        homography_matrix,
        (original_width, original_height)
    )

    return aligned_generated


# ---------------------------------------------------------------------------
# Section 2: Structural Verification
# ---------------------------------------------------------------------------

def verify_structural_integrity(img_original_path, img_generated_path, threshold=0.85):
    """
    Compare the structural integrity of a generated furniture image against
    an original reference image.

    The function aligns the generated image to the original's camera perspective
    using ORB feature matching and homography, then measures structural similarity
    with SSIM.  This prevents false failures caused purely by viewpoint differences
    (different zoom level, camera angle, or slight rotation) rather than genuine
    structural defects.

    PASS examples (same furniture, different angle):
        - Same wardrobe, same door count, same handles, shot from a different angle
        - Same proportions, just more or less zoomed in

    FAIL examples (genuine structural differences):
        - Different number of doors or shelves
        - Missing handles
        - Warped or deformed geometry
        - Significantly different proportions

    Pipeline:
        load → align (ORB + homography) → grayscale → SSIM → evaluate

    Debug outputs written to the logs/ directory:
        - logs/debug_alignment.png   : original | aligned-generated side-by-side
        - logs/diff_map_failed.png   : pixel-level diff map (only on FAIL)

    Parameters
    ----------
    img_original_path : str
        File path to the original / reference furniture image.
    img_generated_path : str
        File path to the AI-generated furniture image to be quality-checked.
    threshold : float, optional
        Minimum SSIM score (0.0–1.0) required for a PASS verdict.
        Default is 0.85. Higher values are stricter.

    Returns
    -------
    tuple[bool, str]
        (True,  "PASS: Structure is consistent.")      if SSIM >= threshold.
        (False, "FAIL: Structure altered! Score: …")   if SSIM <  threshold.
    """

    # ------------------------------------------------------------------ #
    # 1. Load images from disk                                             #
    # ------------------------------------------------------------------ #
    img_original  = cv2.imread(img_original_path)
    img_generated = cv2.imread(img_generated_path)

    # ------------------------------------------------------------------ #
    # 2. Align the generated image to the original's perspective           #
    #                                                                      #
    #    This is the key step that prevents false failures. Before this    #
    #    change, SSIM compared pixels at the same (x, y) position in both #
    #    images. If the camera angle differs, the wardrobe's door edges    #
    #    land on different pixels, producing a low SSIM even though the   #
    #    furniture is structurally identical.                               #
    #                                                                      #
    #    After alignment, img_generated_aligned is a perspective-warped    #
    #    copy of img_generated whose viewpoint matches img_original, so    #
    #    SSIM now compares structurally equivalent pixels.                 #
    # ------------------------------------------------------------------ #
    img_generated_aligned = align_images(img_original, img_generated)

    # ------------------------------------------------------------------ #
    # 3. Save a side-by-side debug image for visual inspection             #
    #    Left panel  = img_original (unchanged reference)                  #
    #    Right panel = img_generated_aligned (after perspective warp)      #
    #    Developers can open this file to verify that alignment looks      #
    #    correct before trusting the SSIM score.                           #
    # ------------------------------------------------------------------ #
    gap = np.full((img_original.shape[0], 40, img_original.shape[2]), 255, dtype=img_original.dtype)
    debug_side_by_side = np.hstack([img_original, gap, img_generated_aligned])
    cv2.imwrite("logs/debug_alignment.png", debug_side_by_side)
    print("[verify_structural_integrity] Debug alignment image saved to "
          "logs/debug_alignment.png")

    # ------------------------------------------------------------------ #
    # 4. Convert to Grayscale                                              #
    #    SSIM is an intensity-based metric. Converting to grayscale        #
    #    removes colour variation that is irrelevant to structural shape   #
    #    comparison and keeps the metric focused on edges and structure.   #
    # ------------------------------------------------------------------ #
    gray_original          = cv2.cvtColor(img_original,          cv2.COLOR_BGR2GRAY)
    gray_generated_aligned = cv2.cvtColor(img_generated_aligned, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------ #
    # 5. Resize if necessary (shapes must match exactly for SSIM)          #
    #    align_images() already outputs the original's exact dimensions,   #
    #    but this resize is kept as a safety net for the fallback path     #
    #    where alignment was skipped and a simple resize was returned.     #
    # ------------------------------------------------------------------ #
    if gray_original.shape != gray_generated_aligned.shape:
        gray_generated_aligned = cv2.resize(
            gray_generated_aligned,
            (gray_original.shape[1], gray_original.shape[0])
        )

    # ------------------------------------------------------------------ #
    # 6. Compute SSIM and get the diff map for dashboards/logs             #
    #    full=True makes ssim() also return a pixel-wise difference image  #
    #    that highlights exactly where the two images diverge.             #
    # ------------------------------------------------------------------ #
    score, diff = ssim(gray_original, gray_generated_aligned, full=True)

    # Scale diff from float [0, 1] to uint8 [0, 255] for image saving.
    diff = (diff * 255).astype("uint8")

    print(f"Structural Similarity Index (SSIM): {score:.4f}")

    # ------------------------------------------------------------------ #
    # 7. Evaluation                                                        #
    # ------------------------------------------------------------------ #
    if score < threshold:
        # Save the diff map so failures are visually inspectable in logs.
        cv2.imwrite("logs/diff_map_failed.png", diff)
        return False, f"FAIL: Structure altered! Score: {score}"

    return True, "PASS: Structure is consistent."
