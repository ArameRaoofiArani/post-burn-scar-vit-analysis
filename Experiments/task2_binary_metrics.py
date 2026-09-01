"""Task 2 -- the same scoring as 04_evaluate_agreement.py, reported with the
0 = scar / 1 = rest label convention.

    python experiments/task2_binary_metrics.py

Identical counts and metrics to 04_evaluate_agreement.py; what differs is the label
convention written to disk and shown on the confusion matrix axes. It also
saves per-slide 0/1 label grids for both the algorithm and the ground
truth, which is what makes it useful as a handover format.

Internally everything is converted to positive = scar before scoring, so
the numbers here and in 04_evaluate_agreement.py cannot disagree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root, for config/metrics

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

import config                                                  # noqa: E402
from config import SCAR_PREDICTION_DIR, SLIDE_NAMES as SLIDE_IDS    # noqa: E402
from metrics import (                                          # noqa: E402
    load_scored_region,
    binary_counts,
    metrics_from_counts,
    sklearn_metrics,
    plot_roc,
    plot_confusion,
)

OUTPUT_DIR = config.OUTPUT_DIR / "metrics_out_task2_binary" / config.EVALUATION

PRED_LABELS = ("Pred: 1 (rest)", "Pred: 0 (scar)")
GT_LABELS = ("GT: 1 (rest)", "GT: 0 (scar)")


def main():
    config.ensure_dirs(OUTPUT_DIR)

    rows = []
    pooled_tp = pooled_fp = pooled_fn = pooled_tn = 0
    pooled_y_true = []
    pooled_y_pred = []
    pooled_y_score = []

    for slide_id in SLIDE_IDS:
        print(f"\n{slide_id}")
        try:
            algo_scar, gt_mask, region_mask, scar_score = load_scored_region(
                slide_id, SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction.npy"
            )
        except FileNotFoundError as e:
            print(f"  Skipping -- missing file: {e}")
            continue

        # 0 = scar, 1 = everything else -- the convention this script reports in
        algo_label = np.where(algo_scar, 0, 1).astype(np.int8)
        gt_label = np.where(gt_mask, 0, 1).astype(np.int8)
        np.save(OUTPUT_DIR / f"{slide_id}_algo_binary_label.npy", algo_label)
        np.save(OUTPUT_DIR / f"{slide_id}_gt_binary_label.npy", gt_label)

        tp, fp, fn, tn = binary_counts(algo_scar, gt_mask, region_mask)
        if tp + fp + fn + tn == 0:
            print("  Skipping -- no tiles in the valid annotated region.")
            continue

        m = metrics_from_counts(tp, fp, fn, tn)
        m.update(sklearn_metrics(algo_scar, gt_mask, region_mask))

        y_true_slide = gt_mask[region_mask].astype(int)     # 1 = scar
        y_pred_slide = algo_scar[region_mask].astype(int)   # 1 = scar
        y_score_slide = scar_score[region_mask]

        m["ROC_AUC"] = plot_roc(y_true_slide, y_score_slide, slide_id,
                                OUTPUT_DIR / f"{slide_id}_roc.png")

        rows.append({"slide_id": slide_id, **m})

        print(f"  tiles compared: {m['n_tiles_compared']}")
        print(f"  accuracy: {m['accuracy']:.3f}  balanced_accuracy: {m['balanced_accuracy']:.3f}  "
              f"sensitivity: {m['sensitivity_recall']:.3f}  specificity: {m['specificity']:.3f}")
        print(f"  precision: {m['precision_ppv']:.3f}  F1: {m['f1_score']:.3f}  "
              f"IoU: {m['IoU']:.3f}  ROC_AUC: {m['ROC_AUC']:.3f}")

        plot_confusion(tp, fp, fn, tn,
                       title=f"{slide_id}\nbinary scar(0) vs rest(1), annotated region only",
                       out_path=OUTPUT_DIR / f"{slide_id}_confusion_matrix_task2.png",
                       pred_labels=PRED_LABELS, gt_labels=GT_LABELS, figsize=(4.5, 4.5))

        pooled_tp += tp; pooled_fp += fp; pooled_fn += fn; pooled_tn += tn
        pooled_y_true.append(y_true_slide)
        pooled_y_pred.append(y_pred_slide)
        pooled_y_score.append(y_score_slide)

    if not rows:
        print("\nNo slides produced results.")
        return

    pooled_m = metrics_from_counts(pooled_tp, pooled_fp, pooled_fn, pooled_tn)

    all_y_true = np.concatenate(pooled_y_true)
    all_y_pred = np.concatenate(pooled_y_pred)
    all_y_score = np.concatenate(pooled_y_score)

    pooled_m["balanced_accuracy"] = balanced_accuracy_score(all_y_true, all_y_pred)
    pooled_m["f1_score"] = f1_score(all_y_true, all_y_pred)
    pooled_m["ROC_AUC"] = plot_roc(all_y_true, all_y_score, "POOLED (all slides)",
                                   OUTPUT_DIR / "POOLED_roc.png")

    rows.append({"slide_id": "POOLED (all slides)", **pooled_m})
    plot_confusion(pooled_tp, pooled_fp, pooled_fn, pooled_tn,
                   title="POOLED\nbinary scar(0) vs rest(1), annotated region only",
                   out_path=OUTPUT_DIR / "POOLED_confusion_matrix_task2.png",
                   pred_labels=PRED_LABELS, gt_labels=GT_LABELS, figsize=(4.5, 4.5))

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "metrics_task2_binary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print("Summary table (label convention: 0 = scar, 1 = rest):")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Confusion matrix plots, ROC curves, and per-slide 0/1 label grids saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
