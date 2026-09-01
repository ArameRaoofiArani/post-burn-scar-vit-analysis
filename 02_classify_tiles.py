"""
Step 1 -- classify every tile and write one scar prediction per slide.

python 02_classify_tiles.py

Reads the per-slide prediction arrays and the five scar sub-cluster
centroids, calls a tile scar when its scar score clears a percentile
threshold AND its nearest centroid is a retained sub-cluster, and saves
<slide>_scar_prediction.npy plus a preview PNG.

Paths come from config.py -- see README.md.
"""

import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
    VALID_CLUSTER_INDICES,
    CLUSTER_COLOURS,
    PCT_THRESHOLD,
)
from metrics import percentile_rank

OUTPUT_DIR = config.SCAR_PREDICTION_DIR
BACKUP_ROOT = OUTPUT_DIR / "scar_prediction_backups"

# --- Operating point -------------------------------------------------------
# Classification rule changes from "argmax==scar AND cluster in {2,3,4}" to
# "percentile_rank(raw scar score, within region) > PCT_THRESHOLD AND
#  cluster in {2,3,4}". 
#
# PCT_THRESHOLD lives in config.py, since the calibration and
# cluster-composition scripts need the identical value. See the comment there
# for how the operating point was chosen.
#
# Baseline (argmax rule) was sensitivity 0.505, specificity 0.953, precision
# 0.950. Only the sensitivities are directly comparable to the threshold arm
# -- the baseline was scored over the bounding-box region and this operating
# point over the connected-component region, and the negative population
# differs between them.

# "valid": rank each tile's percentile within valid_mask (GT-annotated
#          region) -- matches exactly what PCT_THRESHOLD was validated
#          against, but requires <slide>_valid_mask.npy to already exist
#          (i.e. Comparison_GT_ALG.py must have been run at least once).
# "tissue": rank within the full has_tissue extent instead -- works without
#           GT (needed for a genuinely new, unannotated slide).


PERCENTILE_REGION = "valid"

COLOURS = CLUSTER_COLOURS
CMAP = ListedColormap(COLOURS)


def backup_existing_outputs():
    """Copy any existing scar_prediction / preview files into a timestamped
    backup folder before this run overwrites them."""
    existing = list(OUTPUT_DIR.glob("*_scar_prediction.npy")) + \
               list(OUTPUT_DIR.glob("*_scar_prediction_preview.png"))
    if not existing:
        print("No existing scar prediction outputs found to back up (first run).")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"pre_filter_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in existing:
        shutil.copy2(f, backup_dir / f.name)
    print(f"Backed up {len(existing)} existing file(s) to: {backup_dir}")


def build_scar_prediction(slide_name, scar_centres):
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_head_arr.npy")
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_norm_arr.npy")

    rows, cols, _ = head_arr.shape
    has_tissue = np.any(head_arr != 0, axis=2)

    valid_mask = None
    if config.RULE == "argmax":
        # The baseline rule needs no ground truth: it assigns clusters across
        # all tissue and is restricted to the scored region later, at scoring
        # time. So this run can happen before 03_extract_ground_truth.py.
        region_mask = has_tissue
        mask_path = config.valid_mask_path(slide_name)
        if mask_path.exists():
            valid_mask = np.load(mask_path)          # for the preview outline only
    elif PERCENTILE_REGION == "valid":
        mask_path = config.valid_mask_path(slide_name)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"PERCENTILE_REGION='valid' but {mask_path} doesn't exist -- "
                f"run 03_extract_ground_truth.py first, or switch to PERCENTILE_REGION='tissue'."
            )
        valid_mask = np.load(mask_path)
        region_mask = valid_mask & has_tissue
    elif PERCENTILE_REGION == "tissue":
        region_mask = has_tissue
    else:
        raise ValueError(f"Unknown PERCENTILE_REGION: {PERCENTILE_REGION}")

    scar_score = head_arr[..., SCAR_CLASS_INDEX]


    pct_grid = np.full((rows, cols), -1.0, dtype=float)
    pct_grid[region_mask] = percentile_rank(scar_score[region_mask])


    coords = np.argwhere(region_mask)
    embeddings = norm_arr[coords[:, 0], coords[:, 1], :]
    distances = np.linalg.norm(embeddings[:, None, :] - scar_centres[None, :, :], axis=-1)
    nearest_cluster = np.argmin(distances, axis=1)

    cluster_grid = np.full((rows, cols), -1, dtype=int)
    cluster_grid[coords[:, 0], coords[:, 1]] = nearest_cluster
    is_scar_cluster = np.isin(cluster_grid, list(VALID_CLUSTER_INDICES))

    predicted_class = np.argmax(head_arr, axis=2)

    if config.RULE == "argmax":
        positive = (predicted_class == SCAR_CLASS_INDEX) & has_tissue & is_scar_cluster
    else:
        positive = (pct_grid > PCT_THRESHOLD) & is_scar_cluster

    scar_prediction = np.where(positive, cluster_grid + 1, 0).astype(np.int32)

    kept_tiles = int(np.sum(positive))
    excluded_by_cluster = int(np.sum((pct_grid > PCT_THRESHOLD) & ~is_scar_cluster))


    other_is_scar = (predicted_class == SCAR_CLASS_INDEX) & has_tissue
    old_kept_tiles = int(np.sum(other_is_scar & is_scar_cluster))

    valid_mask_used = valid_mask

    return (scar_prediction, int(has_tissue.sum()), kept_tiles, excluded_by_cluster,
            old_kept_tiles, has_tissue, valid_mask_used)


def save_preview(scar_prediction, has_tissue, valid_mask, slide_name):
    from matplotlib.colors import to_rgb

    rows, cols = scar_prediction.shape
    rgb = np.full((rows, cols, 3), 255, dtype=np.uint8)          # background: white
    rgb[has_tissue] = (211, 211, 211)                             # any tissue: light gray

    # Colored scar classification on top (values that actually occur are
    # 3, 4, 5 -- cluster_idx 2,3,4 -- since VALID_CLUSTER_INDICES={2,3,4})
    for value in range(1, 6):
        color = tuple(int(255 * c) for c in to_rgb(COLOURS[value]))
        rgb[scar_prediction == value] = color

    plt.figure(figsize=(6, 12))
    plt.imshow(rgb, interpolation="nearest")
    if valid_mask is not None:
        plt.contour(valid_mask.astype(int), levels=[0.5], colors="blue",
                    linestyles="dashed", linewidths=1)
    rule_text = ("argmax==scar" if config.RULE == "argmax"
                 else f"percentile>{PCT_THRESHOLD:.3f}")
    plt.title(f"{slide_name}\n{rule_text} AND cluster in {sorted(VALID_CLUSTER_INDICES)}\n"
              f"(gray = full tissue extent, dashed line = annotated/scored region)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{slide_name}_scar_prediction_preview.png", dpi=150)
    plt.close()


def main():
    config.ensure_dirs(OUTPUT_DIR)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'  "
          f"(rule: {config.RULE}, region: {config.REGION})")
    print(f"Writing scar predictions to: {OUTPUT_DIR}\n")
    backup_existing_outputs()

    scar_centres = np.load(SCAR_CENTRES_PATH)
    assert scar_centres.shape[0] == 5, (
        f"expected 5 centroids in {SCAR_CENTRES_PATH}, got {scar_centres.shape[0]}"
    )

    summary = []

    for slide_name in SLIDE_NAMES:
        print(f"Processing {slide_name} ...")
        try:
            (scar_prediction, tissue_tiles, kept_scar_tiles, excluded_by_cluster,
             old_kept_tiles, has_tissue, valid_mask_used) = build_scar_prediction(
                slide_name, scar_centres
            )
        except FileNotFoundError as e:
            print(f"  SKIPPED - file not found: {e}")
            summary.append((slide_name, "MISSING FILES", "-", "-", "-"))
            continue

        np.save(OUTPUT_DIR / f"{slide_name}_scar_prediction.npy", scar_prediction)
        save_preview(scar_prediction, has_tissue, valid_mask_used, slide_name)

        pct_scar = 100 * kept_scar_tiles / tissue_tiles if tissue_tiles else 0
        summary.append((slide_name, tissue_tiles, kept_scar_tiles, old_kept_tiles, f"{pct_scar:.1f}%"))
        print(f"  tissue tiles: {tissue_tiles}, kept scar tiles (NEW rule): {kept_scar_tiles} "
              f"({pct_scar:.1f}% scar), kept scar tiles (OLD argmax rule, for comparison): {old_kept_tiles}, "
              f"excluded by cluster despite clearing percentile bar: {excluded_by_cluster}")

    print("\n--- SUMMARY ---")
    print(f"{'slide':<20}{'tissue tiles':<15}{'kept (NEW)':<12}{'kept (OLD)':<12}{'% scar'}")
    for row in summary:
        print(f"{row[0]:<20}{str(row[1]):<15}{str(row[2]):<12}{str(row[3]):<12}{row[4]}")


if __name__ == "__main__":
    main()