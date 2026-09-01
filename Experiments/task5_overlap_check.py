"""Task 5 check -- macro-cluster reassignment, with and without outlier removal.

    python experiments/task5_overlap_check.py

Scores three grids against the ground truth: the Task 1 baseline, the raw
macro-cluster grid, and the macro-cluster grid after IsolationForest
outlier removal.

CAVEAT: the same rule-mismatch warning as task4_overlap_check.py applies to
the baseline column -- <slide>_scar_prediction.npy is now built by the
percentile-threshold rule, not the argmax rule Task 5 was compared against.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for config/metrics

import pandas as pd                                            # noqa: E402

import config                                                  # noqa: E402
from config import SCAR_PREDICTION_DIR, SLIDE_NAMES as SLIDE_IDS    # noqa: E402
from metrics import score_grid, pool_counts, rate_metrics      # noqa: E402

METRIC_NAMES = ["sensitivity", "specificity", "precision", "IoU", "Dice", "accuracy"]

OUTPUT_DIR = config.OUTPUT_DIR / "experiments"


def pool(counts_list):
    return pool_counts(counts_list)


def pooled_metrics(counts):
    return rate_metrics(counts["TP"], counts["FP"], counts["FN"], counts["TN"])


def main():
    config.ensure_dirs(OUTPUT_DIR)
    rows = []
    pooled = {"baseline": [], "task5_raw": [], "task5": []}

    for slide_id in SLIDE_IDS:
        baseline_path = SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction.npy"
        raw_path = SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction_task5_raw.npy"
        after_path = SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction_task5.npy"

        if not (raw_path.exists() and after_path.exists()):
            print(f"{slide_id}: missing Task 5 scar prediction file(s), skipping")
            continue

        raw = score_grid(slide_id, raw_path)
        after = score_grid(slide_id, after_path)
        pooled["task5_raw"].append(raw)
        pooled["task5"].append(after)

        have_baseline = baseline_path.exists()
        baseline = score_grid(slide_id, baseline_path) if have_baseline else None
        if have_baseline:
            pooled["baseline"].append(baseline)

        row = {"slide_id": slide_id}
        for metric in METRIC_NAMES:
            if have_baseline:
                row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_task5_raw"] = raw[metric]
            row[f"{metric}_task5"] = after[metric]
            row[f"{metric}_delta_outliers"] = after[metric] - raw[metric]
        row["removed_TP_by_outliers"] = raw["TP"] - after["TP"]
        row["removed_FP_by_outliers"] = raw["FP"] - after["FP"]
        rows.append(row)

        print(f"\n{slide_id}")
        if have_baseline:
            print(f"  Task 1 baseline:  sens {baseline['sensitivity']:.3f}  "
                  f"prec {baseline['precision']:.3f}  IoU {baseline['IoU']:.3f}")
        print(f"  Task 5 raw:       sens {raw['sensitivity']:.3f}  "
              f"prec {raw['precision']:.3f}  IoU {raw['IoU']:.3f}")
        print(f"  Task 5 (outliers removed): sens {after['sensitivity']:.3f}  "
              f"prec {after['precision']:.3f}  IoU {after['IoU']:.3f}")
        print(f"  TP lost to outlier removal: {raw['TP']-after['TP']}   "
              f"FP lost to outlier removal: {raw['FP']-after['FP']}")

    if not rows:
        print("No slides scored -- check that Task 5 scar prediction files exist.")
        return

    print(f"\n{'='*70}")
    print("POOLED (all slides):")
    for label in ["baseline", "task5_raw", "task5"]:
        if not pooled[label]:
            continue
        pm = pooled_metrics(pool(pooled[label]))
        print(f"\n  [{label}]")
        for metric in METRIC_NAMES:
            print(f"    {metric:<12} {pm[metric]:.3f}")

    if pooled["task5_raw"] and pooled["task5"]:
        raw_counts = pool(pooled["task5_raw"])
        after_counts = pool(pooled["task5"])
        print(f"\n  TP lost to macro-outlier removal (pooled): {raw_counts['TP']-after_counts['TP']}")
        print(f"  FP lost to macro-outlier removal (pooled): {raw_counts['FP']-after_counts['FP']}")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "task5_overlap_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
