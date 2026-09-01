"""Where in embedding space do the missed tiles sit? (Methods 2.5)

    python analysis_missed_scar_location.py

Every tile that is ground-truth positive but called not-scar by the
pipeline is assigned to its nearest of the four macro centroids
(scar_centres_4.npy) by Euclidean distance, to identify which region of
embedding space the misses occupy.

Read analysis_missed_scar_centroid_check.py first: it checks whether that
centroid labelling can be trusted, by applying the same assignment to
confirmed true positives.

The reported figures are from the baseline arm, so run with
SCAR_EVALUATION=baseline.
"""

import csv

import numpy as np

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_PREDICTION_DIR,
    SCAR_CENTRES_4_PATH,
    SLIDE_NAMES,
)
from metrics import load_outcome_masks

OUTPUT_DIR = config.OUTPUT_DIR / "missed_scar_analysis" / config.EVALUATION

OUT_CSV = OUTPUT_DIR / "missed_scar_location.csv"

# Row order in scar_centres_4.npy: 2 = scar, 3 = peri. Rows 0 and 1 were
# never identified with a tissue type.
MACRO_CLUSTER_NAMES = {0: "cluster0_unidentified", 1: "cluster1_unidentified",
                       2: "scar", 3: "peri"}


def head_arr_path(slide):
    return MODEL_OUTPUT_DIR / f"{slide}_scar_head_arr.npy"


def norm_arr_path(slide):
    return MODEL_OUTPUT_DIR / f"{slide}_scar_norm_arr.npy"


# ------------------------------ MAIN ------------------------------------

def main():
    config.ensure_dirs(OUTPUT_DIR)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'\n")
    scar_centres_4 = np.load(SCAR_CENTRES_4_PATH)
    assert scar_centres_4.shape[0] == 4, f"expected 4 rows, got {scar_centres_4.shape[0]}"

    pooled_cluster_counts = {k: 0 for k in range(4)}
    total_fn = 0
    total_gt_pos = 0
    per_slide_rows = []

    for slide in SLIDE_NAMES:
        head_arr = np.load(head_arr_path(slide))          # (rows, cols, 4)
        norm_arr = np.load(norm_arr_path(slide))          # (rows, cols, 1024) -- embeddings

        gt, algo, fn_mask = load_outcome_masks(
            slide, SCAR_PREDICTION_DIR / f"{slide}_scar_prediction.npy")

        assert gt.shape == head_arr.shape[:2] == norm_arr.shape[:2], (
            f"{slide}: shape mismatch -- masks {gt.shape}, "
            f"head_arr {head_arr.shape[:2]}, norm_arr {norm_arr.shape[:2]}. "
            f"These must all be the same tile grid before anything below means anything."
        )
        n_fn = int(fn_mask.sum())
        n_gt_pos = int(gt.sum())
        total_fn += n_fn
        total_gt_pos += n_gt_pos

        if n_fn > 0:
            fn_embeddings = norm_arr[fn_mask]   # (n_fn, 1024)
            dist = np.linalg.norm(
                fn_embeddings[:, None, :] - scar_centres_4[None, :, :], axis=2
            )
            fn_nearest = np.argmin(dist, axis=1)   # (n_fn,)
        else:
            fn_nearest = np.array([], dtype=int)

        slide_cluster_counts = {k: int((fn_nearest == k).sum()) for k in range(4)}
        for k, cnt in slide_cluster_counts.items():
            pooled_cluster_counts[k] += cnt

        per_slide_rows.append({
            "slide": slide,
            "n_fn": n_fn,
            "n_gt_pos": n_gt_pos,
            "fn_rate": round(n_fn / n_gt_pos, 3) if n_gt_pos else None,
            **{f"fn_nearest_{MACRO_CLUSTER_NAMES[k]}": slide_cluster_counts[k] for k in range(4)},
        })

        rate_str = f"{n_fn/n_gt_pos:.1%}" if n_gt_pos else "n/a"
        print(f"{slide}: {n_fn} FN tiles / {n_gt_pos} GT-positive ({rate_str} missed)")
        print("    nearest-macro-centroid breakdown: "
              + ", ".join(f"{MACRO_CLUSTER_NAMES[k]}={slide_cluster_counts[k]}" for k in range(4)))

    print("\n=== POOLED ===")
    if total_gt_pos:
        print(f"Total FN: {total_fn} / {total_gt_pos} GT-positive "
              f"({total_fn/total_gt_pos:.1%} missed, sensitivity = "
              f"{1 - total_fn/total_gt_pos:.3f})")
    for k in range(4):
        pct = pooled_cluster_counts[k] / total_fn * 100 if total_fn else 0
        print(f"  nearest={MACRO_CLUSTER_NAMES[k]}: {pooled_cluster_counts[k]} "
              f"({pct:.1f}% of all missed tiles)")

    fieldnames = list(per_slide_rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_slide_rows)
    print(f"\nSaved per-slide breakdown to {OUT_CSV}")


if __name__ == "__main__":
    main()
