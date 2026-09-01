"""
Step 2 -- turn the QuPath point annotations into ground-truth masks and
score the algorithm against them.

python 03_extract_ground_truth.py

For every slide in the QuPath project this orders the annotation points into
a closed boundary, fills it to a tile grid, saves <slide>_gt_scar_mask.npy
and both region definitions (<slide>_valid_mask_bbox.npy and
<slide>_valid_mask_component.npy), and writes two alignment overlays (real
tissue and black background) with the TP/FP/FN/TN breakdown.

The overlays and counts use whichever region SCAR_EVALUATION selects; the
mask files themselves are written regardless, so this script does not need
re-running when that choice changes.

Requires OpenSlide and a QuPath installation -- see README.md.
"""

import os

import config

# OpenSlide's DLLs (Windows) and QuPath must be locatable before the imports
# below, so these two calls have to come first.
config.configure_openslide()
config.configure_qupath()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from scipy.ndimage import label as cc_label
from openslide import OpenSlide
from PIL import Image
from paquo.projects import QuPathProject

# pixel limit -- safe to disable here, source is trusted and read-only.
Image.MAX_IMAGE_PIXELS = None

from annotations import (
    order_points_annotation,
    two_opt_cleanup,
    get_tiles_inside_boundary,
)
from config import (
    SCAR_PREDICTION_DIR,
    MODEL_OUTPUT_DIR as PRED_DIR,
    SLIDE_DIR,
    SLIDE_NAMES,
    GT_DIR as OUTPUT_DIR,
    QUPATH_PROJECT,
    TILE_SIZE as tile_size,
)

VALID_MASK_PAD_TILES = 5
MIN_GT_OVERLAP_TILES = 5   # a tissue blob needs at least this many GT tiles to count as "the annotated piece"
CROP_PAD_TILES = 10        # extra margin (in tiles) around the kept tissue when cropping the preview


def build_gt_grid(tiles, shape, offset_x=0, offset_y=0):
    rows, cols = shape
    grid = np.zeros(shape, dtype=bool)
    for t in tiles:
        c = (t["tile_x"] + offset_x) // tile_size
        r = (t["tile_y"] + offset_y) // tile_size
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = True
    return grid


def build_valid_mask(gt_grid, pad_tiles=VALID_MASK_PAD_TILES):

    shape = gt_grid.shape
    row_idx, col_idx = np.where(gt_grid)
    if len(row_idx) == 0:
        # No GT tiles at all -- nothing to compare, return an empty mask
        # rather than silently including the whole slide.
        return np.zeros(shape, dtype=bool)

    r_min = max(0, row_idx.min() - pad_tiles)
    r_max = min(shape[0] - 1, row_idx.max() + pad_tiles)
    c_min = max(0, col_idx.min() - pad_tiles)
    c_max = min(shape[1] - 1, col_idx.max() + pad_tiles)

    mask = np.zeros(shape, dtype=bool)
    mask[r_min:r_max + 1, c_min:c_max + 1] = True
    return mask


def restrict_to_tissue_with_gt(has_tissue, gt_grid, min_overlap_tiles=MIN_GT_OVERLAP_TILES):
    """Keep the ENTIRE connected tissue blob(s) (8-connectivity) that overlap
    GT by at least min_overlap_tiles -- the full physical piece of tissue
    that has the doctor's markings on it, not just a tight box around the GT
    points themselves. A second, separate tissue piece with no GT overlap
    is correctly excluded (that's still the point of this function). No
    intersection with a padded bounding box anymore -- that was cutting off
    real parts of the correct tissue piece that happened to sit outside the
    box built from GT point coordinates alone."""
    structure = np.ones((3, 3), dtype=int)
    labeled, n_blobs = cc_label(has_tissue, structure=structure)
    keep = np.zeros_like(has_tissue, dtype=bool)
    for blob_id in range(1, n_blobs + 1):
        blob_mask = labeled == blob_id
        overlap = int(np.sum(blob_mask & gt_grid))
        if overlap >= min_overlap_tiles:
            keep |= blob_mask
    return keep


def overlap_score(a, b):
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return inter / union if union else 0.0


def get_tissue_thumbnail(slide, grid_shape, max_dim=1600):
    rows, cols = grid_shape
    grid_w_px = cols * tile_size
    grid_h_px = rows * tile_size
    scale = min(max_dim / grid_w_px, max_dim / grid_h_px, 1.0)
    thumb_w = max(1, int(round(grid_w_px * scale)))
    thumb_h = max(1, int(round(grid_h_px * scale)))

    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb = thumb.resize((thumb_w, thumb_h), Image.LANCZOS)
    return thumb, thumb_w, thumb_h


def build_confusion_overlay_rgba(tp_mask, fp_mask, fn_mask, tn_mask, alpha):
    """One shared color scheme for both preview variants below:
    TP=yellow, FP=green, FN=red, TN=light blue (all within the scored
    region only -- tiles outside it get no color)."""
    rgba = np.zeros((*tp_mask.shape, 4), dtype=np.uint8)
    rgba[tp_mask] = (255, 255, 0, alpha)
    rgba[fp_mask] = (0, 255, 0, alpha)
    rgba[fn_mask] = (255, 0, 0, alpha)
    rgba[tn_mask] = (0, 200, 255, alpha)
    return rgba


def save_overlay_blackbg(gt_grid, algo_grid, slide_name, label):
    rgb = np.zeros((*gt_grid.shape, 3), dtype=np.uint8)
    rgb[..., 0] = gt_grid.astype(np.uint8) * 255    # red = GT
    rgb[..., 1] = algo_grid.astype(np.uint8) * 255  # green = algo
    plt.figure(figsize=(6, 12))
    plt.imshow(rgb)
    plt.title(f"{slide_name} - {label}\n(red=GT only, green=algo only, yellow=both)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{slide_name}_alignment_{label}_blackbg.png"), dpi=150)
    plt.close()


def save_overlay_real_tissue(tp_mask, fp_mask, fn_mask, tn_mask, tissue_thumb, thumb_w, thumb_h,
                              valid_mask, slide_name, label):
    rows, cols = tp_mask.shape
    overlay_rgba = build_confusion_overlay_rgba(tp_mask, fp_mask, fn_mask, tn_mask, alpha=140)
    overlay_img = Image.fromarray(overlay_rgba, mode="RGBA").resize(
        (thumb_w, thumb_h), Image.NEAREST
    )
    composited = Image.alpha_composite(tissue_thumb.convert("RGBA"), overlay_img)

    plt.figure(figsize=(6, 12))
    plt.imshow(np.array(composited), extent=(0, cols, rows, 0))
    if valid_mask is not None:
        plt.contour(valid_mask.astype(int), levels=[0.5], colors="blue",
                    linestyles="dashed", linewidths=1)
    plt.title(f"{slide_name} - {label}\nTP=yellow  FP=green  FN=red  TN=light blue "
              f"(dashed=scored region)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{slide_name}_alignment_{label}.png"), dpi=150)
    plt.close()


def normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def match_slide_name(image_name):
    norm_image = normalize(image_name)
    for slide_name in SLIDE_NAMES:
        if normalize(slide_name) in norm_image:
            return slide_name
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Ground truth vs. algorithm comparison")
    print("Outputs per slide, written to the ground_truth folder:")
    print("  <slide>_gt_scar_mask.npy     annotated scar, as a tile mask")
    print("  <slide>_valid_mask_bbox.npy       baseline region definition")
    print("  <slide>_valid_mask_component.npy  threshold region definition")
    print("  <slide>_alignment_final.png  overlay on the real tissue image")
    print("  <slide>_alignment_final_blackbg.png   overlay on black")
    print("=" * 60)

    with QuPathProject(str(QUPATH_PROJECT), mode='r') as project:
        for image in project.images:
            name = image.image_name
            print(f"\n{'='*60}\n{name}")

            slide_name = match_slide_name(name)
            if slide_name is None:
                print("  Could not match this QuPath image to a known slide name, skipping.")
                continue

            records = []
            for annotation in image.hierarchy.annotations:
                roi = annotation.roi
                if roi.geom_type == "MultiPoint":
                    for point in roi.geoms:
                        records.append({"x_px": point.x, "y_px": point.y})
            df = pd.DataFrame(records)
            print(f"  Points: {len(df)}")
            if len(df) < 3:
                print("  Not enough points, skipping.")
                continue

            ordered = order_points_annotation(list(zip(df["x_px"], df["y_px"])))
            ordered = two_opt_cleanup(ordered, verbose_name=name)
            tiles = get_tiles_inside_boundary(ordered, tile_size=tile_size)
            print(f"  Tiles extracted: {len(tiles)}")

            scar_prediction = np.load(os.path.join(SCAR_PREDICTION_DIR, f"{slide_name}_scar_prediction.npy"))
            algo_scar_mask = scar_prediction > 0
            shape = scar_prediction.shape

            head_arr = np.load(os.path.join(PRED_DIR, f"{slide_name}_scar_head_arr.npy"))
            has_tissue = np.any(head_arr != 0, axis=-1)

            slide = OpenSlide(os.path.join(str(SLIDE_DIR), image.image_name))
            bounds_x = int(slide.properties.get("openslide.bounds-x", 0))
            bounds_y = int(slide.properties.get("openslide.bounds-y", 0))
            print("  Fetching tissue thumbnail from .mrxs (can take a moment on large slides)...")
            tissue_thumb, thumb_w, thumb_h = get_tissue_thumbnail(slide, shape)
            print(f"  Thumbnail ready: {thumb_w}x{thumb_h}")
            slide.close()

            gt_grid = build_gt_grid(tiles, shape, bounds_x, bounds_y)
            score = overlap_score(gt_grid, algo_scar_mask)
            print(f"  Overlap with algo scar mask, whole slide (bounds offset {bounds_x},{bounds_y}): {score:.3f}")

            region_masks = {
                "bbox": build_valid_mask(gt_grid),
                "component": restrict_to_tissue_with_gt(has_tissue, gt_grid),
            }
            for region_name, mask in region_masks.items():
                np.save(config.valid_mask_path(slide_name, region_name), mask)

            valid_mask = region_masks[config.REGION]
            region_mask = valid_mask & has_tissue
            print("  Region definitions written: "
                  + ", ".join(f"{n} ({int(m.sum())} tiles)"
                              for n, m in region_masks.items())
                  + f"; scoring with '{config.REGION}'")

            masked_score = overlap_score(gt_grid & valid_mask, algo_scar_mask & valid_mask)
            print(f"  Overlap within annotated-tissue region only:          {masked_score:.3f}")

            tp_mask = gt_grid & algo_scar_mask & region_mask
            fp_mask = ~gt_grid & algo_scar_mask & region_mask
            fn_mask = gt_grid & ~algo_scar_mask & region_mask
            tn_mask = ~gt_grid & ~algo_scar_mask & region_mask
            print(f"  TP={int(tp_mask.sum())}  FP={int(fp_mask.sum())}  "
                  f"FN={int(fn_mask.sum())}  TN={int(tn_mask.sum())}")

            np.save(config.gt_mask_path(slide_name), gt_grid)

            save_overlay_blackbg(gt_grid, algo_scar_mask, slide_name, "final")
            save_overlay_real_tissue(tp_mask, fp_mask, fn_mask, tn_mask, tissue_thumb, thumb_w, thumb_h,
                                      valid_mask, slide_name, "final")
            print(f"  Saved: {slide_name}_alignment_final_blackbg.png and {slide_name}_alignment_final.png")
            print(f"  Saved alignment overlay: {slide_name}_alignment_final.png")

    print("\nDone. GT masks saved to the ground_truth folder as <slide>_gt_scar_mask.npy")
    print("Region masks saved as <slide>_valid_mask_<region>.npy, one per region")
    print("definition -- these mark the tissue actually scored, and every downstream")
    print("script uses the one matching SCAR_EVALUATION to exclude un-annotated")
    print("tissue from its metrics and figures.")
    print("Check the *_alignment_final.png files -- should be mostly yellow with some")
    print("red/green fringe (that fringe is real model disagreement, not misalignment).")


if __name__ == "__main__":
    main()
