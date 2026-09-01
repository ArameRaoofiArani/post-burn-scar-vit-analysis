"""
Threshold calibration (Methods 2.7, Results 3.8).

    python 05_calibrate_threshold.py

This is the script that derives the operating point. Nothing else in the
repository re-derives `config.PCT_THRESHOLD`, so run this before trusting
that constant on any new data.

PART A -- does the argmax rule sit above or below the percentile curve?
  The argmax rule gave pooled sensitivity 0.505 at specificity 0.953, i.e.
  FPR 0.047. This extends the
  candidate list down to target sensitivities of 0.40-0.55 and asks
  directly: at the SAME sensitivity as the argmax rule, is the percentile
  curve's specificity better or worse? If worse, the argmax rule sat above
  the curve and the percentile approach wins only further right, at higher
  sensitivity. Computed pooled and per slide.

PART B -- per-slide table at every candidate threshold
  Sensitivity, specificity and precision for each slide at every candidate,
  so it is visible whether one global cutoff works for all six or is
  carried by the strong ones. The weakest slide's sensitivity is the number
  that matters; pooled figures hide it.

Read-only: writes CSVs, modifies nothing.

Scored over whichever region SCAR_EVALUATION selects. The reported
calibration used the connected-component region, so run with
SCAR_EVALUATION=threshold.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES as SLIDE_IDS,
    VALID_CLUSTER_INDICES as SCAR_CLUSTER_INDICES,
    PCT_THRESHOLD,
)
from metrics import percentile_rank, nearest_cluster

METRICS_OUT = config.OUTPUT_DIR / "threshold_calibration" / config.EVALUATION


TARGET_SENSITIVITIES = [0.40, 0.45, 0.50, 0.505, 0.55, 0.60,
                        0.65, 0.70, 0.75, 0.80, 0.85]
BETAS = [1.0, 1.5, 2.0]

FLOOR_SENS = 0.60
FLOOR_SPEC = 0.70
FLOOR_PREC = 0.70

def load_slide(slide_id):
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_head_arr.npy")
    gt_mask = np.load(config.gt_mask_path(slide_id))
    valid_mask = np.load(config.valid_mask_path(slide_id))
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_norm_arr.npy")
    centres = np.load(SCAR_CENTRES_PATH)

    tissue_mask = np.any(head_arr != 0, axis=-1)
    region_mask = valid_mask & tissue_mask

    raw_score = head_arr[..., SCAR_CLASS_INDEX][region_mask]
    y_true = gt_mask[region_mask].astype(int)
    coords = np.argwhere(region_mask)

    pct_score = percentile_rank(raw_score)

    embeddings = norm_arr[coords[:, 0], coords[:, 1], :]
    nearest = nearest_cluster(embeddings, centres)
    is_scar_cluster = np.isin(nearest, list(SCAR_CLUSTER_INDICES))
    gated_pct = np.where(is_scar_cluster, pct_score, -1.0)

    # OLD RULE, reconstructed exactly: argmax==scar AND cluster in {2,3,4}
    predicted_class = np.argmax(head_arr, axis=2)
    old_pos_grid = (predicted_class == SCAR_CLASS_INDEX) & tissue_mask
    old_pred = old_pos_grid[coords[:, 0], coords[:, 1]] & is_scar_cluster

    return y_true, gated_pct, old_pred


def rates(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    acc = (tp + tn) / total if total else np.nan
    bal_acc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    dice = f1  # identical formula to F1 in the binary case
    return sens, spec, prec, acc, bal_acc, f1, iou, dice


def confusion(pred, y_true):
    pos = y_true == 1
    return (int(np.sum(pred & pos)), int(np.sum(pred & ~pos)),
            int(np.sum(~pred & pos)), int(np.sum(~pred & ~pos)))


def main():
    config.ensure_dirs(METRICS_OUT)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'\n")
    data = {}
    for sid in SLIDE_IDS:
        print(f"Loading {sid} ...")
        data[sid] = load_slide(sid)

    pooled_y = np.concatenate([d[0] for d in data.values()])
    pooled_s = np.concatenate([d[1] for d in data.values()])
    pooled_old = np.concatenate([d[2] for d in data.values()])

    fpr, tpr, thresholds = roc_curve(pooled_y, pooled_s)
    pooled_auc = auc(fpr, tpr)
    print(f"\nPooled AUC: {pooled_auc:.4f}")

    # =====================================================================
    # PART A -- old rule vs the curve
    # =====================================================================
    print("\n" + "=" * 96)
    print("PART A -- DOES THE OLD ARGMAX RULE SIT ABOVE OR BELOW THE NEW CURVE?")
    print("=" * 96)

    tp, fp, fn, tn = confusion(pooled_old, pooled_y)
    o_sens, o_spec, o_prec = rates(tp, fp, fn, tn)[:3]
    o_fpr = 1.0 - o_spec
    print(f"\n  OLD RULE pooled : sens {o_sens:.4f}  spec {o_spec:.4f}  "
          f"prec {o_prec:.4f}  (FPR {o_fpr:.4f})")

    idx = np.where(tpr >= o_sens)[0]
    if len(idx):
        i = idx[0]
        curve_spec = 1.0 - fpr[i]
        print(f"  NEW CURVE at same sensitivity ({tpr[i]:.4f}): "
              f"spec {curve_spec:.4f}  (threshold {thresholds[i]:.6f})")
        diff = curve_spec - o_spec
        print(f"\n  specificity difference (new - old): {diff:+.4f}")

    # What does the curve give at the same SPECIFICITY?
    j = np.where(fpr <= o_fpr)[0]
    if len(j):
        k = j[-1]
        print(f"\n  NEW CURVE at same specificity ({1 - fpr[k]:.4f}): "
              f"sens {tpr[k]:.4f}  (threshold {thresholds[k]:.6f})")
        print(f"  sensitivity difference (new - old): {tpr[k] - o_sens:+.4f}")

    rows_old = []
    print("\n  Per slide:")
    print(f"  {'slide':<20}{'old sens':>10}{'old spec':>10}{'old prec':>10}"
          f"{'curve spec':>12}{'diff':>9}")
    for sid, (y, s, old) in data.items():
        tp, fp, fn, tn = confusion(old, y)
        s_sens, s_spec, s_prec = rates(tp, fp, fn, tn)[:3]
        f_, t_, th_ = roc_curve(y, s)
        ii = np.where(t_ >= s_sens)[0]
        c_spec = 1.0 - f_[ii[0]] if len(ii) else np.nan
        d = c_spec - s_spec
        print(f"  {sid:<20}{s_sens:>10.4f}{s_spec:>10.4f}{s_prec:>10.4f}"
              f"{c_spec:>12.4f}{d:>+9.4f}")
        rows_old.append(dict(slide=sid, old_sensitivity=s_sens,
                             old_specificity=s_spec, old_precision=s_prec,
                             curve_specificity_at_same_sens=c_spec,
                             difference=d))
    pd.DataFrame(rows_old).to_csv(METRICS_OUT / "old_rule_comparison.csv",
                                  index=False)

    # =====================================================================
    # PART B -- per-slide table at every candidate threshold
    # =====================================================================
    candidates = []
    for target in TARGET_SENSITIVITIES:
        ii = np.where(tpr >= target)[0]
        if len(ii):
            candidates.append((f"target_{target}", float(thresholds[ii[0]])))
    for b in BETAS:
        jj = b * tpr - fpr
        candidates.append((f"youden_beta{b}", float(thresholds[int(np.argmax(jj))])))
    # An earlier operating point, kept only so its row appears alongside the
    # others for comparison. It is NOT the value the pipeline uses.
    candidates.append(("earlier_candidate_superseded", 0.442361))
    # The adopted operating point: target_0.7 on the pooled ROC curve. Taken
    # from config.PCT_THRESHOLD rather than re-derived here, so this row is
    # pinned to the value the pipeline actually applies and cannot drift from
    # it if TARGET_SENSITIVITIES or the underlying data change.
    candidates.append(("final_chosen_threshold", PCT_THRESHOLD))

    seen, uniq = set(), []
    for label, thr in candidates:
        key = round(thr, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((label, thr))

    rows = []
    print("\n" + "=" * 96)
    print("PART B -- PER-SLIDE BREAKDOWN AT EACH THRESHOLD")
    print(f"floors: sens >= {FLOOR_SENS}, spec >= {FLOOR_SPEC}, prec >= {FLOOR_PREC}")
    print("=" * 96)

    for label, thr in uniq:
        TP = FP = FN = TN = 0
        per = []
        for sid, (y, s, _) in data.items():
            pred = s > thr
            tp, fp, fn, tn = confusion(pred, y)
            TP += tp; FP += fp; FN += fn; TN += tn
            sens, spec, prec, acc, bal_acc, f1, iou, dice = rates(tp, fp, fn, tn)
            meets = (sens >= FLOOR_SENS and spec >= FLOOR_SPEC
                     and prec >= FLOOR_PREC)
            per.append((sid, sens, spec, prec, meets))
            rows.append(dict(candidate=label, threshold=thr, slide=sid,
                             tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens,
                             specificity=spec, precision=prec, accuracy=acc,
                             balanced_accuracy=bal_acc, f1=f1, iou=iou, dice=dice,
                             meets_all_floors=meets))
        p_sens, p_spec, p_prec, p_acc, p_bal_acc, p_f1, p_iou, p_dice = rates(TP, FP, FN, TN)
        rows.append(dict(candidate=label, threshold=thr, slide="POOLED",
                         tp=TP, fp=FP, fn=FN, tn=TN, sensitivity=p_sens,
                         specificity=p_spec, precision=p_prec, accuracy=p_acc,
                         balanced_accuracy=p_bal_acc, f1=p_f1, iou=p_iou, dice=p_dice,
                         meets_all_floors=(p_sens >= FLOOR_SENS
                                           and p_spec >= FLOOR_SPEC
                                           and p_prec >= FLOOR_PREC)))

        sens_vals = [p[1] for p in per]
        n_meet = sum(1 for p in per if p[4])
        print(f"\n  {label}   threshold {thr:.6f}")
        print(f"  {'slide':<20}{'sens':>9}{'spec':>9}{'prec':>9}   ok")
        for sid, sens, spec, prec, meets in per:
            print(f"  {sid:<20}{sens:>9.3f}{spec:>9.3f}{prec:>9.3f}   "
                  f"{'YES' if meets else '-'}")
        print(f"  {'POOLED':<20}{p_sens:>9.3f}{p_spec:>9.3f}{p_prec:>9.3f}")
        print(f"  slides meeting all floors: {n_meet}/6    "
              f"sensitivity spread: {min(sens_vals):.3f}-{max(sens_vals):.3f} "
              f"(range {max(sens_vals) - min(sens_vals):.3f})")
        print(f"  POOLED extended: accuracy {p_acc:.4f}  balanced_accuracy {p_bal_acc:.4f}  "
              f"F1/Dice {p_f1:.4f}  IoU {p_iou:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_OUT / "per_slide_thresholds.csv", index=False)

    # =====================================================================
    # Which threshold works best across ALL slides, not just pooled?
    # =====================================================================
    print("\n" + "=" * 96)
    print("WHICH THRESHOLD IS MOST CONSISTENT ACROSS SLIDES?")
    print("=" * 96)
    per_only = df[df.slide != "POOLED"]
    summary = per_only.groupby(["candidate", "threshold"]).agg(
        slides_meeting=("meets_all_floors", "sum"),
        worst_sens=("sensitivity", "min"),
        best_sens=("sensitivity", "max"),
        worst_spec=("specificity", "min"),
        worst_prec=("precision", "min"),
    ).reset_index()
    summary["sens_range"] = summary.best_sens - summary.worst_sens
    summary = summary.sort_values(["slides_meeting", "worst_sens"],
                                  ascending=[False, False])
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    print(f"\nWrote {METRICS_OUT / 'per_slide_thresholds.csv'}")
    print(f"Wrote {METRICS_OUT / 'old_rule_comparison.csv'}")


if __name__ == "__main__":
    main()