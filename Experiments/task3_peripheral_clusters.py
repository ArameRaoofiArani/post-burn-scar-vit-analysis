"""Task 3 -- does adding one peripheral cluster recover missed scar?

    python experiments/task3_peripheral_clusters.py

Keeps the Task 1 scar calls unchanged and additionally marks (as value 6)
every tile whose head class is "peri" and whose nearest peri centroid is
PERIPHERAL_CLUSTER_INDEX. The point is to see whether the model's peri
class is hiding scar the scar channel missed.

Needs peri_centres_4.npy, which the main pipeline does not use.

Exploratory: this writes <slide>_scar_prediction_task3.npy and never touches the
baseline grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for config/metrics

import shutil                                                # noqa: E402
from datetime import datetime                                  # noqa: E402

import numpy as np                                             # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import ListedColormap                   # noqa: E402

import config                                                  # noqa: E402
from config import (                                           # noqa: E402
    MODEL_OUTPUT_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
    VALID_CLUSTER_INDICES,                  # from Task 1, unchanged
)

PERI_CENTRES_PATH = config.PROJECT_DIR / "peri_centres_4.npy"
PERI_CLASS_INDEX = 3                        # the "peri" head channel
PERIPHERAL_CLUSTER_INDEX = 3                # peri_centres_4 cluster under test

OUTPUT_DIR = config.SCAR_PREDICTION_DIR
BACKUP_ROOT = OUTPUT_DIR / "scar_prediction_backups"

# index 0..6, position = tile value
COLOURS = ["black", "khaki", "limegreen", "teal", "royalblue", "orangered", "magenta"]
CMAP = ListedColormap(COLOURS)


def backup_existing_outputs():
    existing = list(OUTPUT_DIR.glob("*_scar_prediction_task3.npy")) + \
               list(OUTPUT_DIR.glob("*_scar_prediction_task3_preview.png"))
    if not existing:
        print("No existing Task 3 scar prediction outputs found to back up (first run).")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"task3_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in existing:
        shutil.copy2(f, backup_dir / f.name)
    print(f"Backed up {len(existing)} existing file(s) to: {backup_dir}")


def build_task3_grid(slide_name, scar_centres, peri_centres):
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_head_arr.npy")
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_norm_arr.npy")

    rows, cols, _ = head_arr.shape
    has_tissue = np.any(head_arr != 0, axis=2)
    predicted_class = np.argmax(head_arr, axis=2)

    scar_prediction = np.zeros((rows, cols), dtype=np.int32)

    is_scar_candidate = (predicted_class == SCAR_CLASS_INDEX) & has_tissue
    scar_idx = np.argwhere(is_scar_candidate)
    n_task1_scar = 0
    if len(scar_idx) > 0:
        scar_embeddings = norm_arr[scar_idx[:, 0], scar_idx[:, 1], :]
        scar_dist = np.linalg.norm(scar_embeddings[:, None, :] - scar_centres[None, :, :], axis=2)
        scar_nearest = np.argmin(scar_dist, axis=1)
        is_task1_scar = np.isin(scar_nearest, list(VALID_CLUSTER_INDICES))
        values = np.where(is_task1_scar, scar_nearest + 1, 0)
        scar_prediction[scar_idx[:, 0], scar_idx[:, 1]] = values
        n_task1_scar = int(is_task1_scar.sum())

    is_peri_candidate = (predicted_class == PERI_CLASS_INDEX) & has_tissue
    peri_idx = np.argwhere(is_peri_candidate)
    n_peripheral_added = 0
    if len(peri_idx) > 0:
        peri_embeddings = norm_arr[peri_idx[:, 0], peri_idx[:, 1], :]
        peri_dist = np.linalg.norm(peri_embeddings[:, None, :] - peri_centres[None, :, :], axis=2)
        peri_nearest = np.argmin(peri_dist, axis=1)
        is_peripheral_added = peri_nearest == PERIPHERAL_CLUSTER_INDEX
        scar_prediction[peri_idx[:, 0], peri_idx[:, 1]] = np.where(is_peripheral_added, 6, 0)
        n_peripheral_added = int(is_peripheral_added.sum())

    n_tissue = int(has_tissue.sum())
    return scar_prediction, n_tissue, n_task1_scar, n_peripheral_added


def save_preview(scar_prediction, valid_mask, slide_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))

    axes[0].imshow(scar_prediction, cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
    axes[0].set_title(f"{slide_name}\nTask 3 grid (magenta=6=peripheral addition)")

    axes[1].imshow(scar_prediction, cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
    if valid_mask is not None:
        axes[1].contour(valid_mask, levels=[0.5], colors="white", linewidths=1.5)
    axes[1].set_title(f"{slide_name}\nsame, with valid_mask boundary (white)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{slide_name}_scar_prediction_task3_preview.png", dpi=150)
    plt.close(fig)


def main():
    config.ensure_dirs(OUTPUT_DIR)
    backup_existing_outputs()

    scar_centres = np.load(SCAR_CENTRES_PATH)
    peri_centres = np.load(PERI_CENTRES_PATH)
    assert scar_centres.shape[0] == 5, f"expected 5 rows in {SCAR_CENTRES_PATH}"
    assert peri_centres.shape[0] == 4, f"expected 4 rows in {PERI_CENTRES_PATH}"

    summary = []

    for slide_name in SLIDE_NAMES:
        print(f"Processing {slide_name} ...")
        try:
            scar_prediction, n_tissue, n_task1_scar, n_peripheral_added = build_task3_grid(
                slide_name, scar_centres, peri_centres
            )
        except FileNotFoundError as e:
            print(f"  SKIPPED - file not found: {e}")
            summary.append((slide_name, "MISSING FILES", "-", "-"))
            continue

        np.save(OUTPUT_DIR / f"{slide_name}_scar_prediction_task3.npy", scar_prediction)

        valid_mask_path = config.valid_mask_path(slide_name)
        valid_mask = np.load(valid_mask_path) if valid_mask_path.exists() else None

        save_preview(scar_prediction, valid_mask, slide_name)

        pct_added = 100 * n_peripheral_added / n_tissue if n_tissue else 0
        summary.append((slide_name, n_tissue, n_task1_scar, n_peripheral_added, f"+{pct_added:.1f}%"))

        # how many of the newly-added peripheral tiles fall inside vs
        # outside valid_mask -- inside means Task 1 was under-detecting
        # scar even within the annotated strip; outside means it's
        # genuinely extending into the un-annotated peripheral strip.
        if valid_mask is not None:
            added_mask = scar_prediction == 6
            added_inside = int((added_mask & valid_mask).sum())
            added_outside = int((added_mask & ~valid_mask).sum())
            print(f"  tissue tiles: {n_tissue}, Task 1 scar: {n_task1_scar}, "
                  f"peripheral addition: {n_peripheral_added} "
                  f"({added_inside} inside valid_mask, {added_outside} outside)")
        else:
            print(f"  tissue tiles: {n_tissue}, Task 1 scar: {n_task1_scar}, "
                  f"peripheral addition: {n_peripheral_added} (no valid_mask found)")

    print("\n--- SUMMARY ---")
    print(f"{'slide':<20}{'tissue tiles':<15}{'task1 scar':<12}{'peripheral added':<18}{'% added'}")
    for row in summary:
        print(f"{row[0]:<20}{str(row[1]):<15}{str(row[2]):<12}{str(row[3]):<18}{row[4] if len(row) > 4 else ''}")

    print("\nDone. Check *_scar_prediction_task3_preview.png for each slide:")
    print("  Left panel: values 3/4/5 = Task 1 scar clusters, 6 (magenta) = peri cluster 2 addition")
    print("  Right panel: same, with white valid_mask boundary overlaid")
    print("This is exploratory. There is no task3_overlap_check.py -- score these grids")
    print("by pointing metrics.score_grid at <slide>_scar_prediction_task3.npy.")


if __name__ == "__main__":
    main()
