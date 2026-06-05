# ============================================================
#  constants.py
#  All project settings live here.
#  To change anything — thresholds, paths, image size —
#  just edit this file. No other file needs to be touched.
# ============================================================

import os

# ------------------------------------------------------------------
# PATHS
# Base directory = the folder where this file lives
# ------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

ORIGINALS_DIR   = os.path.join(BASE_DIR, "images", "originals")   # Original furniture images
GENERATED_DIR   = os.path.join(BASE_DIR, "images", "generated")   # AI-generated output images
LOGS_DIR        = os.path.join(BASE_DIR, "logs")                   # Text logs
DIFF_MAPS_DIR   = os.path.join(BASE_DIR, "logs", "diff_maps")      # Visual diff images on failure
LOG_FILE        = os.path.join(BASE_DIR, "logs", "qa_results.log") # Log file path

# ------------------------------------------------------------------
# IMAGE PREPROCESSING
# Both images are resized to the same dimensions before comparing.
# Larger = more detail but slower. 512x512 is a good balance.
# ------------------------------------------------------------------
TARGET_WIDTH        = 512   # Resize width in pixels
TARGET_HEIGHT       = 512   # Resize height in pixels
CONVERT_TO_GRAYSCALE = True  # True = compare structure only (ignore color)

# ------------------------------------------------------------------
# SSIM SETTINGS
# SSIM (Structural Similarity Index) scores from -1 to 1.
# 1.0 = identical structure, 0.0 = no similarity, -1.0 = inverted
# ------------------------------------------------------------------
SSIM_THRESHOLD  = 0.85  # Below this score = FAIL (structure was altered)
SSIM_WIN_SIZE   = 7     # Sliding window size — must be an odd number (3, 5, 7, 11)
SAVE_DIFF_MAP   = True  # Save a visual diff image to logs/ when a test fails

# ------------------------------------------------------------------
# FSIM SETTINGS (optional — more robust than SSIM for lighting changes)
# Disabled by default due to Python 3.12 library incompatibility.
# Set FSIM_ENABLED = True only if you install image-similarity-measures.
# ------------------------------------------------------------------
FSIM_ENABLED    = False  # Set to True to also run FSIM comparison
FSIM_THRESHOLD  = 0.80   # Below this score = FAIL (only used if FSIM_ENABLED)

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
LOG_TO_FILE     = True   # Also write results to LOG_FILE path above
LOG_LEVEL       = "INFO" # Options: "DEBUG", "INFO", "WARNING", "ERROR"


# ---- Quick test: run this file directly to verify paths are correct ----
if __name__ == "__main__":
    print("✅ Constants loaded successfully!")
    print(f"   Project root   : {BASE_DIR}")
    print(f"   Originals dir  : {ORIGINALS_DIR}")
    print(f"   Generated dir  : {GENERATED_DIR}")
    print(f"   Diff maps dir  : {DIFF_MAPS_DIR}")
    print(f"   SSIM threshold : {SSIM_THRESHOLD}")
    print(f"   Image size     : {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"   FSIM enabled   : {FSIM_ENABLED}")