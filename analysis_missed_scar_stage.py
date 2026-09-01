"""Where in the pipeline are the missed tiles lost? (Methods 2.5)

    python analysis_missed_scar_stage.py

Splits the false negatives by the stage that discarded them:

  Stage A  candidate selection -- was the tile in the argmax == scar
           population at all? If not, it never reached sub-clustering.
  Stage B  for tiles that WERE candidates: which of the five
           scar_centres_5 sub-clusters is nearest? Kept = {2, 3, 4},
           excluded = {0, 1}.

This separates the classifier's contribution to missed sensitivity from the
sub-cluster filter's.

The reported figures are from the baseline arm, so run with
SCAR_EVALUATION=baseline.
"""

import csv

import numpy as np

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_PREDICTION_DIR,
    SCAR_CENTRES_5_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
    VALID_CLUSTER_INDICES,
)
from metrics import load_outcome_masks

OUTPUT_DIR = config.OUTPUT_DIR / "missed_scar_analysis" / config.EVALUATION
OUT_CSV = OUTPUT_DIR / "missed_scar_stage.csv"

EXCLUDED_CLUSTER_INDICES = set(range(5)) - VALID_CLUSTER_INDICES


def head_arr_path(slide):
    return MODEL_OUTPUT_DIR / f"{slide}_scar_head_arr.npy"


def norm_arr_path(slide):
    return MODEL_OUTPUT_DIR / f"{slide}_scar_norm_arr.npy"


# ------------------------------ MAIN ------------------------------------

def main():
    config.ensure_dirs(OUTPUT_DIR)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'\n")
    scar_centres_5 = np.load(SCAR_CENTRES_5_PATH)
    assert scar_centres_5.shape[0] == 5, f"expected 5 rows, got {scar_centres_5.shape[0]}"

    total_fn = 0
    total_gt_pos = 0
    total_not_candidate = 0          # FN tiles head_arr never even calls scar
    pooled_subcluster_counts = {k: 0 for k in range(5)}
    per_slide_rows = []

    for slide in SLIDE_NAMES:
        head_arr = np.load(head_arr_path(slide))
        norm_arr = np.load(norm_arr_path(slide))

        gt, algo, fn_mask = load_outcome_masks(
            slide, SCAR_PREDICTION_DIR / f"{slide}_scar_prediction.npy")

        assert gt.shape == head_arr.shape[:2] == norm_arr.shape[:2], \
            f"{slide}: shape mismatch across masks/head_arr/norm_arr"

        n_fn = int(fn_mask.sum())
        n_gt_pos = int(gt.sum())
        total_fn += n_fn
        total_gt_pos += n_gt_pos

        # Stage A: candidate selection, same as compute_task1_baseline()
        has_tissue = np.any(head_arr != 0, axis=2)
        predicted_class = np.argmax(head_arr, axis=2)
        is_scar_candidate = (predicted_class == SCAR_CLASS_INDEX) & has_tissue

        fn_is_candidate = fn_mask & is_scar_candidate
        n_not_candidate = n_fn - int(fn_is_candidate.sum())
        total_not_candidate += n_not_candidate

        # Stage B: sub-cluster nearest-centroid, only for FN tiles that ARE candidates
        slide_subcluster_counts = {k: 0 for k in range(5)}
        if fn_is_candidate.sum() > 0:
            embeddings = norm_arr[fn_is_candidate]
            dist = np.linalg.norm(embeddings[:, None, :] - scar_centres_5[None, :, :], axis=2)
            nearest = np.argmin(dist, axis=1)
            for k in range(5):
                cnt = int((nearest == k).sum())
                slide_subcluster_counts[k] = cnt
                pooled_subcluster_counts[k] += cnt

        n_kept = sum(slide_subcluster_counts[k] for k in VALID_CLUSTER_INDICES)
        n_excluded = sum(slide_subcluster_counts[k] for k in EXCLUDED_CLUSTER_INDICES)

        per_slide_rows.append({
            "slide": slide,
            "n_fn": n_fn,
            "n_gt_pos": n_gt_pos,
            "fn_not_head_arr_candidate": n_not_candidate,
            "fn_candidate_kept_subcluster": n_kept,
            "fn_candidate_excluded_subcluster": n_excluded,
            **{f"fn_subcluster_{k}": slide_subcluster_counts[k] for k in range(5)},
        })

        print(f"{slide}: {n_fn} FN tiles")
        print(f"    not even a head_arr scar-candidate: {n_not_candidate}")
        print(f"    candidate, but nearest sub-cluster EXCLUDED {sorted(EXCLUDED_CLUSTER_INDICES)}: {n_excluded}")
        print(f"    candidate, but nearest sub-cluster KEPT {sorted(VALID_CLUSTER_INDICES)} "
              f"(unexpected -- should be algo=1, check for stale scar_prediction.npy): {n_kept}")
        print(f"    sub-cluster breakdown: {slide_subcluster_counts}")

    print("\n=== POOLED ===")
    print(f"Total FN: {total_fn} / {total_gt_pos} GT-positive")
    print(f"  not a head_arr scar-candidate at all: {total_not_candidate} "
          f"({100*total_not_candidate/total_fn:.1f}%)")
    pooled_kept = sum(pooled_subcluster_counts[k] for k in VALID_CLUSTER_INDICES)
    pooled_excluded = sum(pooled_subcluster_counts[k] for k in EXCLUDED_CLUSTER_INDICES)
    print(f"  candidate, nearest sub-cluster EXCLUDED {sorted(EXCLUDED_CLUSTER_INDICES)}: {pooled_excluded} "
          f"({100*pooled_excluded/total_fn:.1f}% of all FN)")
    print(f"  candidate, nearest sub-cluster KEPT {sorted(VALID_CLUSTER_INDICES)}: {pooled_kept} "
          f"({100*pooled_kept/total_fn:.1f}% of all FN -- should be near 0; if not, "
          f"scar_prediction.npy on disk may be stale relative to scar_centres_5.npy)")
    print(f"  full sub-cluster breakdown (pooled): {pooled_subcluster_counts}")

    fieldnames = list(per_slide_rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_slide_rows)
    print(f"\nSaved per-slide breakdown to {OUT_CSV}")


if __name__ == "__main__":
    main()
