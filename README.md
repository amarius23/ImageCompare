# ImageComp — Furniture QA: Image Structural Integrity Checker

A quality assurance tool that automatically compares AI-generated furniture images against original reference photos to detect structural deformations — missing doors, wrong proportions, warped geometry, missing handles, etc.

## What it does

Given a pair of images (an original product photo and an AI-generated version), the tool:

1. **Detects the furniture region** automatically using GPT-4o Vision, rembg (U2-Net), or Sobel+Otsu edge detection — ignoring background objects like lamps, plants, and artwork.
2. **Aligns the images** using LoFTR (transformer-based) or ORB feature matching to correct for camera angle and zoom differences before comparing.
3. **Runs a multi-metric scoring stack:**
   - **SSIM** — structural similarity on Canny edge maps masked to the furniture region
   - **LPIPS** — perceptual distance using a pretrained AlexNet
   - **dHash** — gradient fingerprint sensitive to structural changes but robust to lighting
   - **FSIM** — optional, disabled by default (Python 3.12 incompatibility)
4. **Saves diff maps and comparison images** to `logs/diff_maps/` for visual inspection.
5. **Logs results** to `logs/qa_results.log`.

**Verdict:** `PASS` if all enabled metrics pass their thresholds. `FAIL` with a diff heatmap showing exactly where the structure diverged.

## Setup

```bash
# 1. Clone the repo and create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
# Note: first run downloads the U2-Net model (~170 MB) to ~/.u2net/ automatically
```

> **Optional:** To enable GPT-4o bounding box detection, set your OpenAI API key in `preprocessor.py`:
> ```python
> OPENAI_API_KEY = "sk-..."
> ```

## Add your images

Place your images in:
```
images/
  originals/   ← reference product photos
  generated/   ← AI-generated versions to check
```

Then register the pairs in the `IMAGE_PAIRS` list at the bottom of `preprocessor.py`:

```python
IMAGE_PAIRS = [
    {
        "name":       "my_cabinet",
        "original":   "images/originals/my_cabinet.jpg",
        "generated":  "images/generated/my_cabinet.jpg",
        "auto_bbox":  True,   # auto-detect furniture region, ignore background
        "debug_bbox": True,   # save a debug image showing the detected box
    },
]
```

## Run

```bash
python preprocessor.py
```

Results are printed to the terminal and written to `logs/qa_results.log`. Visual diff maps and comparison panels are saved to `logs/diff_maps/`.

## Configuration

All thresholds and feature flags are at the top of `preprocessor.py`:

| Setting | Default | Description |
|---|---|---|
| `SSIM_THRESHOLD` | `0.85` | Minimum SSIM score to pass (strict mode) |
| `LPIPS_THRESHOLD` | `0.45` | Maximum perceptual distance to pass |
| `HASH_THRESHOLD` | `15` | Maximum Hamming distance to pass |
| `REMBG_ENABLED` | `True` | Use U2-Net for background removal |
| `LOFTR_ENABLED` | `True` | Use LoFTR for image alignment |
| `OPENAI_BBOX_ENABLED` | `True` | Use GPT-4o to locate furniture |
| `SAVE_DIFF_MAP` | `True` | Always save diff heatmaps |

Set `SSIM_STRICT_MODE = False` to use a looser threshold of `0.75` for images shot in different rooms.

## Output

```
logs/
  qa_results.log          ← timestamped pass/fail log
  diff_maps/
    <name>_comparison_*.png   ← ORIGINAL | GENERATED | DIFF side-by-side
    <name>_diff_*.png         ← heatmap (red = structural problem)
    bbox_debug_*.png          ← detected bounding boxes (if debug_bbox=True)
    alignment_debug_*.png     ← aligned image pairs
```
