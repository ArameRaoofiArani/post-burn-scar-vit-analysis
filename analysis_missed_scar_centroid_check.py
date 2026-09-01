"""Validity check on the macro-centroid labelling (Methods 2.5).

    python analysis_missed_scar_centroid_check.py

analysis_missed_scar_location.py assigns missed tiles to their nearest of the
four macro centroids and reads index 2 as "looks like scar". That reading
is only meaningful if the correspondence holds in the first place.

This applies the identical assignment to confirmed TRUE POSITIVES -- tiles
where the ground truth says scar and the pipeline agrees. If those do not
land on centroid 2 at close to 100%, the labelling is unreliable and the
false-negative result cannot be read as "these tiles look like scar".

Run this before trusting analysis_missed_scar_location.py, and with the same
SCAR_EVALUATION setting.
"""

import numpy as np

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_PREDICTION_DIR,
    SCAR_CENTRES_4_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
)
from metrics import load_outcome_masks


def main():
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'\n")
    scar_centres_4 = np.load(SCAR_CENTRES_4_PATH)

    total_tp = 0
    total_tp_nearest_scar = 0

    for slide in SLIDE_NAMES:
        norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide}_scar_norm_arr.npy")

        gt, algo, _ = load_outcome_masks(
            slide, SCAR_PREDICTION_DIR / f"{slide}_scar_prediction.npy")
        tp_mask = gt & algo   # GT says scar, pipeline agrees -- confident, unambiguous scar tiles

        n_tp = int(tp_mask.sum())
        total_tp += n_tp

        if n_tp > 0:
            embeddings = norm_arr[tp_mask]
            dist = np.linalg.norm(embeddings[:, None, :] - scar_centres_4[None, :, :], axis=2)
            nearest = np.argmin(dist, axis=1)
            n_nearest_scar = int((nearest == SCAR_CLASS_INDEX).sum())
        else:
            n_nearest_scar = 0

        total_tp_nearest_scar += n_nearest_scar
        pct = 100 * n_nearest_scar / n_tp if n_tp else float("nan")
        print(f"{slide}: {n_tp} confirmed-TP scar tiles, {n_nearest_scar} ({pct:.1f}%) "
              f"nearest macro cluster 2")

    print("\n=== POOLED ===")
    pct = 100 * total_tp_nearest_scar / total_tp if total_tp else float("nan")
    print(f"{total_tp} confirmed-TP scar tiles, {total_tp_nearest_scar} ({pct:.1f}%) "
          f"nearest macro cluster 2")
    print("\nHigher percentages indicate stronger correspondence between "
          "macro-centroid assignment and classifier output.")


if __name__ == "__main__":
    main()
