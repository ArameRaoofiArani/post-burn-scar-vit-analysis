"""Step 3 -- score the classifier against the ground truth.

    python 04_evaluate_agreement.py

Per slide and pooled across slides: TP/FP/FN/TN, accuracy, sensitivity,
specificity, precision, IoU, Dice, balanced accuracy, F1 and ROC AUC, all
computed inside the scored region only. Writes metrics.csv plus a
confusion matrix and ROC curve per slide.

Run after 02_classify_tiles.py and 03_extract_ground_truth.py.
"""

import numpy as np
import pandas as pd

import config
from config import SCAR_PREDICTION_DIR, SLIDE_NAMES as SLIDE_IDS
from metrics import (
    load_scored_region,
    binary_counts,
    metrics_from_counts,
    sklearn_metrics,
    plot_roc,
    plot_confusion,
)
from sklearn.metrics import balanced_accuracy_score, f1_score

OUTPUT_DIR = config.OUTPUT_DIR / "metrics_out" / config.EVALUATION


def main():
    config.ensure_dirs(OUTPUT_DIR)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'  "
          f"(rule: {config.RULE}, region: {config.REGION})")

    rows = []
    pooled_tp = pooled_fp = pooled_fn = pooled_tn = 0
    pooled_y_true = []
    pooled_y_score = []
    pooled_algo_scar = []
    pooled_gt_mask = []

    for slide_id in SLIDE_IDS:
        print(f"\n{slide_id}")
        try:
            algo_scar, gt_mask, region_mask, raw_scar_score = load_scored_region(
                slide_id, SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction.npy"
            )
        except FileNotFoundError as e:
            print(f"  Skipping -- missing file: {e}")
            continue

        tp, fp, fn, tn = binary_counts(algo_scar, gt_mask, region_mask)
        if tp + fp + fn + tn == 0:
            print("  Skipping -- no tiles in the valid annotated region (check valid_mask).")
            continue

        m = metrics_from_counts(tp, fp, fn, tn)
        extra = sklearn_metrics(algo_scar, gt_mask, region_mask)

        y_true_slide = gt_mask[region_mask].astype(int)
        y_score_slide = raw_scar_score[region_mask]
        extra["roc_auc"] = plot_roc(y_true_slide, y_score_slide, slide_id,
                                    OUTPUT_DIR / f"{slide_id}_roc.png")

        rows.append({"slide_id": slide_id, **m, **extra})

        print(f"  tiles compared: {m['n_tiles_compared']}")
        print(f"  accuracy: {m['accuracy']:.3f}  sensitivity: {m['sensitivity_recall']:.3f}  "
              f"specificity: {m['specificity']:.3f}")
        print(f"  precision: {m['precision_ppv']:.3f}  IoU: {m['IoU']:.3f}")
        print(f"  balanced_accuracy: {extra['balanced_accuracy']:.3f}  "
              f"f1_score: {extra['f1_score']:.3f}  roc_auc: {extra['roc_auc']:.3f}")

        plot_confusion(tp, fp, fn, tn,
                       title=f"{slide_id}\n(annotated tissue region only)",
                       out_path=OUTPUT_DIR / f"{slide_id}_confusion_matrix.png")

        pooled_tp += tp; pooled_fp += fp; pooled_fn += fn; pooled_tn += tn
        pooled_y_true.append(y_true_slide)
        pooled_y_score.append(y_score_slide)
        pooled_algo_scar.append(algo_scar[region_mask])
        pooled_gt_mask.append(gt_mask[region_mask])

    if not rows:
        print("\nNo slides produced results -- nothing to save. "
              "Make sure 03_extract_ground_truth.py has been (re)run "
              "so *_gt_scar_mask.npy and *_valid_mask.npy exist for all slides.")
        return

    pooled_m = metrics_from_counts(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

    pooled_algo_flat = np.concatenate(pooled_algo_scar)
    pooled_gt_flat = np.concatenate(pooled_gt_mask)
    pooled_extra = {
        "balanced_accuracy": balanced_accuracy_score(pooled_gt_flat.astype(int),
                                                     pooled_algo_flat.astype(int)),
        "f1_score": f1_score(pooled_gt_flat.astype(int), pooled_algo_flat.astype(int)),
    }
    pooled_extra["roc_auc"] = plot_roc(
        np.concatenate(pooled_y_true), np.concatenate(pooled_y_score),
        "POOLED (all slides)", OUTPUT_DIR / "POOLED_roc.png",
    )

    rows.append({"slide_id": "POOLED (all slides)", **pooled_m, **pooled_extra})
    plot_confusion(pooled_tp, pooled_fp, pooled_fn, pooled_tn,
                   title="POOLED\n(annotated tissue region only)",
                   out_path=OUTPUT_DIR / "POOLED_confusion_matrix.png")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("Summary table:")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Confusion matrix + ROC plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
