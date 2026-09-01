"""Task 4 check -- did removing sub-cluster outliers actually help?

    python experiments/task4_overlap_check.py

Scores the baseline scar prediction and the outlier-removed one from
task4_outlier_subclusters.py against the ground truth, and reports the
change in each metric per slide and pooled.

CAVEAT: "before" here is whatever <slide>_scar_prediction.npy currently holds.
That file is now written by the percentile-threshold rule in
02_classify_tiles.py, while Task 4 was developed against the older
argmax rule. Comparing the two mixes a change of classification rule into
what is meant to be an outlier-removal delta. Regenerate both grids under
the same rule before reading anything into these numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for config/metrics

import pandas as pd                                            # noqa: E402

import config                                                  # noqa: E402
from config import SCAR_PREDICTION_DIR, SLIDE_NAMES as SLIDE_IDS    # noqa: E402
from metrics import score_grid, rate_metrics                    # noqa: E402

METRIC_NAMES = ["sensitivity", "specificity", "precision", "IoU", "Dice", "accuracy"]


OUTPUT_DIR = config.OUTPUT_DIR / "experiments"


def main():
    config.ensure_dirs(OUTPUT_DIR)
    rows = []
    pooled_before = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    pooled_after = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for slide_id in SLIDE_IDS:
        before_path = SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction.npy"
        after_path = SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction_task4.npy"

        if not (before_path.exists() and after_path.exists()):
            print(f"{slide_id}: missing scar prediction file(s), skipping")
            continue

        before = score_grid(slide_id, before_path)
        after = score_grid(slide_id, after_path)

        for k in pooled_before:
            pooled_before[k] += before[k]
            pooled_after[k] += after[k]

        row = {"slide_id": slide_id}
        for metric in METRIC_NAMES:
            row[f"{metric}_before"] = before[metric]
            row[f"{metric}_after"] = after[metric]
            row[f"{metric}_delta"] = after[metric] - before[metric]
        row["removed_TP"] = before["TP"] - after["TP"]
        row["removed_FP"] = before["FP"] - after["FP"]
        rows.append(row)

        print(f"\n{slide_id}")
        print(f"  IoU:         {before['IoU']:.3f} -> {after['IoU']:.3f}  "
              f"(delta {after['IoU']-before['IoU']:+.3f})")
        print(f"  sensitivity: {before['sensitivity']:.3f} -> {after['sensitivity']:.3f}  "
              f"(delta {after['sensitivity']-before['sensitivity']:+.3f})")
        print(f"  precision:   {before['precision']:.3f} -> {after['precision']:.3f}  "
              f"(delta {after['precision']-before['precision']:+.3f})")
        print(f"  TP lost by removing outliers: {before['TP']-after['TP']}")
        print(f"  FP lost by removing outliers: {before['FP']-after['FP']}")

    if not rows:
        print("No slides scored -- check that both scar prediction files exist.")
        return

    def m(counts):
        return rate_metrics(counts["TP"], counts["FP"], counts["FN"], counts["TN"])

    pooled_before_m = m(pooled_before)
    pooled_after_m = m(pooled_after)

    print(f"\n{'='*60}")
    print("POOLED (all slides):")
    for metric in METRIC_NAMES:
        b, a = pooled_before_m[metric], pooled_after_m[metric]
        print(f"  {metric:<12} {b:.3f} -> {a:.3f}  (delta {a-b:+.3f})")
    print(f"  TP lost by removing outliers: {pooled_before['TP'] - pooled_after['TP']}")
    print(f"  FP lost by removing outliers: {pooled_before['FP'] - pooled_after['FP']}")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "task4_outlier_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
