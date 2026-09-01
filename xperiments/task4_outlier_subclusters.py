"""Task 4 -- prune sub-cluster outliers with an IsolationForest.

    python experiments/task4_outlier_subclusters.py

Reproduces the Task 1 (argmax) scar grid, then fits one IsolationForest
per retained sub-cluster on the pooled embeddings of every slide and drops
the tiles it flags. The question is whether scattered, embedding-space
anomalies inside the scar clusters are false positives worth removing.

Exploratory: writes <slide>_scar_prediction_task4.npy. Score it with
task4_overlap_check.py, and read that script's caveat first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for config/metrics

import shutil                                                # noqa: E402
from datetime import datetime                                  # noqa: E402
from collections import defaultdict                            # noqa: E402

import numpy as np                                             # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import ListedColormap                   # noqa: E402
from sklearn.ensemble import IsolationForest                   # noqa: E402

import config                                                  # noqa: E402
from config import (                                           # noqa: E402
    MODEL_OUTPUT_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
    VALID_CLUSTER_INDICES,                  # from Task 1, unchanged
)

CONTAMINATION = "auto"   # sklearn default heuristic; tune to a float (e.g. 0.05) if needed
RANDOM_STATE = 0         # fixed seed for reproducibility

OUTPUT_DIR = config.SCAR_PREDICTION_DIR
BACKUP_ROOT = OUTPUT_DIR / "scar_prediction_backups"

# 0=not scar, 3/4/5=clusters 2/3/4 (kept), 7=removed-as-outlier (for preview only)
COLOURS = ["black", "khaki", "limegreen", "teal", "royalblue", "orangered", "magenta", "red"]
CMAP = ListedColormap(COLOURS)


def backup_existing_outputs():
    existing = list(OUTPUT_DIR.glob("*_scar_prediction_task4.npy")) + \
               list(OUTPUT_DIR.glob("*_scar_prediction_task4_preview.png"))
    if not existing:
        print("No existing Task 4 scar prediction outputs found to back up (first run).")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / f"task4_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in existing:
        shutil.copy2(f, backup_dir / f.name)
    print(f"Backed up {len(existing)} existing file(s) to: {backup_dir}")


def compute_task1_scar(slide_name, scar_centres):
    """Reproduces Task 1's scar prediction exactly, and also returns the
    embeddings + coordinates for every scar tile, so outlier detection
    can be fit on them afterward."""
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_head_arr.npy")
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_norm_arr.npy")

    has_tissue = np.any(head_arr != 0, axis=2)
    predicted_class = np.argmax(head_arr, axis=2)
    is_scar_candidate = (predicted_class == SCAR_CLASS_INDEX) & has_tissue

    scar_idx = np.argwhere(is_scar_candidate)
    if len(scar_idx) == 0:
        return np.zeros(head_arr.shape[:2], dtype=np.int32), {}

    embeddings = norm_arr[scar_idx[:, 0], scar_idx[:, 1], :]
    dist = np.linalg.norm(embeddings[:, None, :] - scar_centres[None, :, :], axis=2)
    nearest = np.argmin(dist, axis=1)
    is_scar = np.isin(nearest, list(VALID_CLUSTER_INDICES))

    scar_prediction = np.zeros(head_arr.shape[:2], dtype=np.int32)
    scar_prediction[scar_idx[is_scar, 0], scar_idx[is_scar, 1]] = nearest[is_scar] + 1

    per_cluster = defaultdict(list)
    for i in np.where(is_scar)[0]:
        r, c = scar_idx[i]
        per_cluster[int(nearest[i])].append((int(r), int(c), embeddings[i]))

    return scar_prediction, per_cluster


def save_preview(scar_prediction_before, scar_prediction_after, valid_mask, slide_name):
    removed = (scar_prediction_before > 0) & (scar_prediction_after == 0)
    display = scar_prediction_after.copy()
    display[removed] = 7

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))

    axes[0].imshow(display, cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
    axes[0].set_title(f"{slide_name}\nred=removed as outlier, others=kept Task 1 sub-clusters")

    axes[1].imshow(display, cmap=CMAP, vmin=0, vmax=7, interpolation="nearest")
    if valid_mask is not None:
        axes[1].contour(valid_mask, levels=[0.5], colors="white", linewidths=1.5)
    axes[1].set_title(f"{slide_name}\nsame, with valid_mask boundary (white)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{slide_name}_scar_prediction_task4_preview.png", dpi=150)
    plt.close(fig)


def main():
    config.ensure_dirs(OUTPUT_DIR)
    backup_existing_outputs()

    scar_centres = np.load(SCAR_CENTRES_PATH)
    assert scar_centres.shape[0] == 5, f"expected 5 rows in {SCAR_CENTRES_PATH}"

    print("Pass 1/2: computing Task 1 scar grids + collecting embeddings per cluster...")
    per_slide_prediction = {}
    pooled_by_cluster = defaultdict(list)   # cluster_idx -> [(slide, r, c, embedding), ...]

    for slide_name in SLIDE_NAMES:
        try:
            scar_prediction, per_cluster = compute_task1_scar(slide_name, scar_centres)
        except FileNotFoundError as e:
            print(f"  {slide_name}: SKIPPED - file not found: {e}")
            continue
        per_slide_prediction[slide_name] = scar_prediction
        for cluster_idx, records in per_cluster.items():
            for (r, c, emb) in records:
                pooled_by_cluster[cluster_idx].append((slide_name, r, c, emb))

    print("\nPass 2/2: fitting IsolationForest per cluster (pooled across all slides)...")
    outlier_flags = defaultdict(set)   # (slide, r, c) -> flagged as outlier

    for cluster_idx in sorted(pooled_by_cluster.keys()):
        records = pooled_by_cluster[cluster_idx]
        X = np.stack([r[3] for r in records])
        print(f"  cluster {cluster_idx}: {len(records)} tiles pooled across all slides")

        clf = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
        preds = clf.fit_predict(X)   # -1 = outlier, 1 = inlier
        n_outliers = int(np.sum(preds == -1))
        pct = 100 * n_outliers / len(records)
        print(f"    flagged as outliers: {n_outliers} ({pct:.1f}%)")

        for (slide_name, r, c, _), pred in zip(records, preds):
            if pred == -1:
                outlier_flags[slide_name].add((r, c))

    print("\nBuilding cleaned scar predictions...")
    summary = []
    for slide_name in SLIDE_NAMES:
        if slide_name not in per_slide_prediction:
            summary.append((slide_name, "MISSING FILES", "-", "-"))
            continue

        prediction_before = per_slide_prediction[slide_name]
        prediction_after = prediction_before.copy()

        flags = outlier_flags.get(slide_name, set())
        for (r, c) in flags:
            prediction_after[r, c] = 0

        np.save(OUTPUT_DIR / f"{slide_name}_scar_prediction_task4.npy", prediction_after)

        # NOTE: deliberately does NOT write a baseline grid if one is missing.
        # prediction_before here is the old argmax rule; writing it to
        # <slide>_scar_prediction.npy would silently replace the pipeline's
        # percentile-threshold output with a different classifier.
        baseline_path = OUTPUT_DIR / f"{slide_name}_scar_prediction.npy"
        if not baseline_path.exists():
            print(f"  {slide_name}: no baseline scar_prediction.npy -- run 02_classify_tiles.py "
                  f"to create one; not writing an argmax-rule grid in its place")

        valid_mask_path = config.valid_mask_path(slide_name)
        valid_mask = np.load(valid_mask_path) if valid_mask_path.exists() else None
        save_preview(prediction_before, prediction_after, valid_mask, slide_name)

        n_before = int(np.sum(prediction_before > 0))
        n_after = int(np.sum(prediction_after > 0))
        n_removed = n_before - n_after
        pct_removed = 100 * n_removed / n_before if n_before else 0
        summary.append((slide_name, n_before, n_after, n_removed, f"{pct_removed:.1f}%"))
        print(f"  {slide_name}: {n_before} -> {n_after} scar tiles ({n_removed} removed, {pct_removed:.1f}%)")

    print("\n--- SUMMARY ---")
    print(f"{'slide':<20}{'before':<10}{'after':<10}{'removed':<10}{'% removed'}")
    for row in summary:
        print(f"{row[0]:<20}{str(row[1]):<10}{str(row[2]):<10}{str(row[3]):<10}{row[4] if len(row) > 4 else ''}")

    print("\nDone. Check *_scar_prediction_task4_preview.png -- red tiles were removed as outliers.")
    print("Sanity check: do removed tiles look like scattered noise / isolated points within")
    print("the scar region, rather than a solid contiguous chunk being stripped out?")
    print("Next: run task4_overlap_check.py to see whether removing outliers improved metrics.")


if __name__ == "__main__":
    main()