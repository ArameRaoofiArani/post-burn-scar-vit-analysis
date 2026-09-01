"""
Composition of the five sub-clusters across the scored region (Methods 2.9).

    python analysis_cluster_composition.py

Produces the numbers behind Table 3. Every tile in the scored region is
assigned to its nearest of the five centroids, independently of any decision
rule, and the composition is reported per slide and pooled.

For each slide and pooled, breaks down cluster membership (0-4) by:
    all region tiles          -- the background distribution
    GT-positive (real scar)   -- where real scar actually sits
    GT-negative               -- where non-scar tissue sits
    TP / FP / FN / TN         -- under BOTH decision rules

BOTH RULES are reported side by side, because "before applying the
threshold" is ambiguous:
    OLD rule : argmax(head_arr) == scar AND cluster in {2,3,4}
    NEW rule : gated percentile > THRESHOLD

Note on the old rule: 100% of its true positives are in {2,3,4} BY
CONSTRUCTION -- the cluster gate is part of the rule. So a figure like 56%
can only be the share in cluster 2 SPECIFICALLY, with the rest in 3 and 4.
The GT-positive row is the one that is not circular: it shows where real
scar sits regardless of any decision rule.

Also reports, per cluster:
    enrichment = P(cluster | GT-positive) / P(cluster | all region tiles)
    scar purity = fraction of tiles in that cluster that are real scar

READ-ONLY. Writes one CSV, modifies nothing.

Output: metrics_out/cluster_composition.csv
"""

import numpy as np
import pandas as pd

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES as SLIDE_IDS,
    VALID_CLUSTER_INDICES as SCAR_CLUSTER_INDICES,
    PCT_THRESHOLD as THRESHOLD,
)
from metrics import percentile_rank, nearest_cluster

METRICS_OUT = config.OUTPUT_DIR / "cluster_composition" / config.EVALUATION

N_CLUSTERS = 5


def load_slide(slide_id):
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_head_arr.npy")
    gt_mask = np.load(config.gt_mask_path(slide_id)).astype(bool)
    valid_mask = np.load(config.valid_mask_path(slide_id)).astype(bool)
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_norm_arr.npy")
    centres = np.load(SCAR_CENTRES_PATH)

    has_tissue = np.any(head_arr != 0, axis=-1)
    region_mask = valid_mask & has_tissue
    coords = np.argwhere(region_mask)

    raw_score = head_arr[..., SCAR_CLASS_INDEX][region_mask]
    y_true = gt_mask[region_mask]

    cluster = nearest_cluster(
        norm_arr[coords[:, 0], coords[:, 1], :], centres)
    is_scar_cluster = np.isin(cluster, list(SCAR_CLUSTER_INDICES))

    # NEW rule
    pct = percentile_rank(raw_score)
    gated = np.where(is_scar_cluster, pct, -1.0)
    pred_new = gated > THRESHOLD

    # OLD rule: argmax == scar AND cluster in {2,3,4}
    predicted_class = np.argmax(head_arr, axis=2)
    argmax_scar = (predicted_class == SCAR_CLASS_INDEX) & has_tissue
    pred_old = argmax_scar[coords[:, 0], coords[:, 1]] & is_scar_cluster

    return dict(cluster=cluster, y_true=y_true,
                pred_old=pred_old, pred_new=pred_new)


def counts_by_cluster(cluster, sel):
    return np.bincount(cluster[sel], minlength=N_CLUSTERS)


def pct_row(counts):
    tot = counts.sum()
    return counts / tot * 100.0 if tot else np.zeros(N_CLUSTERS)


def print_block(title, rows, note=None):
    print(f"\n  {title}")
    if note:
        print(f"  {note}")
    print(f"  {'category':<24}{'n':>9}   " +
          "".join(f"{'c' + str(i):>9}" for i in range(N_CLUSTERS)))
    for label, counts in rows:
        p = pct_row(counts)
        print(f"  {label:<24}{counts.sum():>9}   " +
              "".join(f"{p[i]:>8.1f}%" for i in range(N_CLUSTERS)))


def analyse(name, cluster, y_true, pred_old, pred_new, csv_rows):
    print(f"\n{'=' * 92}")
    print(f"{name}")
    print(f"{'=' * 92}")

    all_sel = np.ones_like(y_true, dtype=bool)
    base = counts_by_cluster(cluster, all_sel)
    gtpos = counts_by_cluster(cluster, y_true)
    gtneg = counts_by_cluster(cluster, ~y_true)

    print_block(
        "GROUND TRUTH -- independent of any decision rule",
        [("all region tiles", base),
         ("GT-positive (real scar)", gtpos),
         ("GT-negative", gtneg)])

    for rule_name, pred in (("OLD rule (argmax==scar AND cluster in 2,3,4)", pred_old),
                            (f"NEW rule (gated percentile > {THRESHOLD:.6f})", pred_new)):
        tp = counts_by_cluster(cluster, pred & y_true)
        fp = counts_by_cluster(cluster, pred & ~y_true)
        fn = counts_by_cluster(cluster, ~pred & y_true)
        tn = counts_by_cluster(cluster, ~pred & ~y_true)
        n_tp, n_fp, n_fn = tp.sum(), fp.sum(), fn.sum()
        sens = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else np.nan
        prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else np.nan
        print_block(rule_name,
                    [("TP", tp), ("FP", fp), ("FN", fn), ("TN", tn)],
                    note=f"sens {sens:.3f}  prec {prec:.3f}")

    # ---- per-cluster enrichment and purity ------------------------------
    print("\n  PER-CLUSTER SUMMARY")
    print(f"  {'cluster':<9}{'region tiles':>14}{'% of region':>13}"
          f"{'GT scar in it':>15}{'scar purity':>13}{'enrichment':>12}   in gate")
    prev = y_true.sum() / len(y_true)
    for c in range(N_CLUSTERS):
        n_c = base[c]
        n_scar = gtpos[c]
        purity = n_scar / n_c if n_c else np.nan
        enrich = (purity / prev) if (n_c and prev) else np.nan
        print(f"  {c:<9}{n_c:>14}{100.0 * n_c / base.sum():>12.1f}%"
              f"{n_scar:>15}{100.0 * purity:>12.1f}%{enrich:>12.2f}   "
              f"{'YES' if c in SCAR_CLUSTER_INDICES else '-'}")
        csv_rows.append(dict(
            scope=name, cluster=c, region_tiles=int(n_c),
            gt_scar_tiles=int(n_scar), scar_purity=purity,
            enrichment=enrich, in_scar_gate=(c in SCAR_CLUSTER_INDICES)))

    # ---- the specific question ------------------------------------------
    gate_gt = gtpos[list(SCAR_CLUSTER_INDICES)].sum()
    print(f"\n  Real scar inside the gate {sorted(SCAR_CLUSTER_INDICES)}: "
          f"{gate_gt} of {gtpos.sum()} = {100.0 * gate_gt / max(gtpos.sum(), 1):.1f}%")
    print(f"  Real scar LOST to clusters 0/1 (unreachable at any threshold): "
          f"{gtpos[0] + gtpos[1]} tiles = "
          f"{100.0 * (gtpos[0] + gtpos[1]) / max(gtpos.sum(), 1):.1f}% of all scar")
    if gate_gt:
        share = 100.0 * gtpos[2] / gate_gt
        print(f"  Of scar inside the gate, cluster 2 holds {share:.1f}% "
              f"(3: {100.0 * gtpos[3] / gate_gt:.1f}%, "
              f"4: {100.0 * gtpos[4] / gate_gt:.1f}%)")
    n_tp_old = (pred_old & y_true).sum()
    if n_tp_old:
        tp_old = counts_by_cluster(cluster, pred_old & y_true)
        print(f"  Of OLD-rule TPs, cluster 2 holds "
              f"{100.0 * tp_old[2] / n_tp_old:.1f}% "
              f"(3: {100.0 * tp_old[3] / n_tp_old:.1f}%, "
              f"4: {100.0 * tp_old[4] / n_tp_old:.1f}%)")


def main():
    config.ensure_dirs(METRICS_OUT)
    print("=" * 92)
    print(f"analysis_cluster_composition.py   SCAR_EVALUATION = '{config.EVALUATION}'")
    print(f"scar gate = {sorted(SCAR_CLUSTER_INDICES)},  "
          f"new-rule threshold = {THRESHOLD:.6f}")
    print("=" * 92)

    data = {}
    for sid in SLIDE_IDS:
        print(f"Loading {sid} ...")
        try:
            data[sid] = load_slide(sid)
        except FileNotFoundError as e:
            print(f"  SKIP - {e}")

    csv_rows = []
    for sid, d in data.items():
        analyse(sid, d["cluster"], d["y_true"], d["pred_old"], d["pred_new"],
                csv_rows)

    cluster = np.concatenate([d["cluster"] for d in data.values()])
    y_true = np.concatenate([d["y_true"] for d in data.values()])
    pred_old = np.concatenate([d["pred_old"] for d in data.values()])
    pred_new = np.concatenate([d["pred_new"] for d in data.values()])
    analyse("POOLED", cluster, y_true, pred_old, pred_new, csv_rows)

    out = METRICS_OUT / "cluster_composition.csv"
    pd.DataFrame(csv_rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
