"""Outlier removal, evaluated end to end (Methods 2.4.2, Results 3.3 and 3.5).

    python analysis_outlier_removal.py

The complete version of the outlier-removal experiment. The exploratory
scripts in Experiments/ produce ground-truth-scored grids at
contamination="auto" only; this runs every (level x method x contamination)
arm end to end, so each yields scar predictions, GT metrics, silhouette, and the
SPATIAL STRUCTURE of the removed tiles. It also produces the six-panel
removal figure.

    largest_frac    fraction of removed tiles in the single biggest
                    connected component. ~1.0 = one dense core;
                    near 0 = scattered speckle.
    singleton_frac  fraction of removed tiles that are isolated, with no
                    removed neighbour. High = scattered noise.
    frac_inside_gt  fraction of removed tiles falling inside the
                    ground-truth scar region.
    nbr_frac        mean fraction of the 8 neighbours that were kept
                    scar, for removed vs kept tiles. If these are close,
                    removal is spatially indiscriminate.

A RANDOM-removal control arm is included. Deleting points and recomputing
silhouette raises it mechanically (extreme points carry low silhouette),
so the honest comparison for any detector is against removing the same
fraction at random -- not against the baseline.

NOTE: the random control is a baseline for the silhouette comparison rather
than a separate analysis. Disable it by removing "random" from METHODS.

SAFETY: writes only inside its own timestamped output folder. Existing
*_scar_prediction*.npy and *_comparison.csv files are never touched.
"""

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
import config                                                    # noqa: E402
from config import (                                             # noqa: E402
    MODEL_OUTPUT_DIR,
    SCAR_CENTRES_4_PATH,          # macro: 4-way
    SCAR_CENTRES_5_PATH,          # sub: 5-way
    SCAR_CLASS_INDEX,             # head_arr channel gating scar candidates
    SLIDE_NAMES as SLIDE_IDS,
    VALID_CLUSTER_INDICES as SUB_KEPT_INDICES,
)

MACRO_SCAR_INDEX = 2          # which of the 4 macro centroids is scar

# Narrow these when you only need to regenerate figures, e.g.
#     METHODS = ["isolation_forest"]
#     CONTAMINATION_VALUES = ["auto"]
# LOF is the slow arm -- it fits per cluster on the full pooled population.
METHODS = ["isolation_forest", "lof", "random"]
CONTAMINATION_VALUES = ["auto", 0.05, 0.1]

SUBSAMPLE_N = 6000       # silhouette subsample, matches the earlier diagnostic
RANDOM_STATE = 0
MIN_CLUSTER_SIZE = 20    # too few points to fit a detector meaningfully

FIGURE_ARMS = [
    ("sub", "isolation_forest", "auto"),
    ("macro", "isolation_forest", "auto"),
]

OUTPUT_DIR = (config.OUTPUT_DIR / "outlier_removal" / config.EVALUATION
              / f"full_eval_{datetime.now():%Y%m%d_%H%M%S}")

# ---------------------------------------------------------------------------
# FIGURE STYLE
# ---------------------------------------------------------------------------
# White background rather than black: journals print on white, and a
# black field wastes ink, darkens the whole plate, and makes the thin
# scattered markers harder to resolve.
BACKGROUND_COLOUR = "#FFFFFF"
TISSUE_COLOUR = "#DEDEDE"     # tissue that is not predicted scar
KEPT_COLOUR = "#7FA9CC"       # retained scar prediction
REMOVED_COLOUR = "#C8102E"    # excluded tile -- strong red, maximum contrast
                              # against both the blue and the grey. (No orange
                              # anywhere in this palette: in the original task4
                              # preview a red "removed" sat beside an orangered
                              # "kept", which is how a dense KEPT region got
                              # read as a dense REMOVED one.)
GT_LINE_COLOUR = "#111111"    # ground-truth scar boundary
REGION_LINE_COLOUR = "#E08214"  # evaluation-region boundary (dashed)

FIG_DPI = 300
PANEL_HEIGHT_IN = 8.0         # display height of each panel
CROP_PAD_TILES = 6
DETAIL_PAD_TILES = 4          # padding around the scored region in panel B

# Tissue-mask cleanup. head_arr.sum() != 0 picks up single-tile-wide streaks
# at scan boundaries and small label/edge blocks; drawn as tissue they appear
# as thin stair-stepped outlines that read as deliberate annotation.
TISSUE_OPEN_ITER = 1          # erode/dilate passes that snap thin streaks off
TISSUE_MIN_COMPONENT = 150    # drop tissue blobs smaller than this many tiles

LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 13
SUPTITLE_FONTSIZE = 15


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_slide_context():
    """Per-slide masks needed for scoring, loaded once and reused by every
    arm. region_mask matches task4/task5_overlap_check.py exactly."""
    ctx = {}
    for slide_id in SLIDE_IDS:
        head_path = MODEL_OUTPUT_DIR / f"{slide_id}_scar_head_arr.npy"
        if not head_path.exists():
            print(f"  {slide_id}: SKIPPED - {head_path.name} not found")
            continue
        head_arr = np.load(head_path)
        tissue_mask = head_arr.sum(axis=-1) != 0

        gt_path = config.gt_mask_path(slide_id)
        valid_path = config.valid_mask_path(slide_id)
        if not (gt_path.exists() and valid_path.exists()):
            print(f"  {slide_id}: SKIPPED - ground-truth mask(s) not found")
            continue
        gt_mask = np.load(gt_path).astype(bool)
        valid_mask = np.load(valid_path).astype(bool)

        ctx[slide_id] = {
            "shape": head_arr.shape[:2],
            "region_mask": valid_mask & tissue_mask,
            "tissue_mask": tissue_mask,
            "gt_mask": gt_mask,
            "valid_mask": valid_mask,
        }
    return ctx


def pool_scar_candidates(ctx):
    """Pools every scar-candidate tile across slides, KEEPING its
    (slide, row, col) so removals can be written back to the grid."""
    embeddings, slide_of, rows, cols = [], [], [], []
    for slide_id in SLIDE_IDS:
        if slide_id not in ctx:
            continue
        head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_head_arr.npy")
        norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_norm_arr.npy")

        has_tissue = np.any(head_arr != 0, axis=2)
        predicted_class = np.argmax(head_arr, axis=2)
        is_candidate = (predicted_class == SCAR_CLASS_INDEX) & has_tissue

        r, c = np.where(is_candidate)
        if len(r) == 0:
            continue
        embeddings.append(norm_arr[r, c, :])
        slide_of.append(np.full(len(r), slide_id, dtype=object))
        rows.append(r)
        cols.append(c)
        print(f"  {slide_id}: {len(r):,} scar-candidate tiles")

    return (np.concatenate(embeddings, axis=0),
            np.concatenate(slide_of),
            np.concatenate(rows),
            np.concatenate(cols))


# ---------------------------------------------------------------------------
# CLUSTERING / OUTLIER FLAGGING
# ---------------------------------------------------------------------------
def assign_clusters(embeddings, centres):
    """cdist rather than broadcast-and-norm: the broadcast allocates an
    (n_tiles, n_centres, n_dims) intermediate, gigabytes at ~94k tiles."""
    return np.argmin(cdist(embeddings, centres), axis=1)


def baseline_scar_prediction(level, labels):
    """Per-tile scar call under the level's baseline rule. 0 = not scar.
    Mirrors build_scar_predictions_task4/task5."""
    prediction = np.zeros(len(labels), dtype=np.int32)
    if level == "sub":
        keep = np.isin(labels, list(SUB_KEPT_INDICES))
        prediction[keep] = labels[keep] + 1          # task4: prediction = nearest + 1
    else:
        keep = labels == MACRO_SCAR_INDEX
        prediction[keep] = 1                          # task5: single scar class
    return prediction


def flag_outliers(embeddings, labels, method, contamination, rng):
    """Fits per cluster, returns a boolean inlier mask over all tiles.

    NOTE on contamination: at a fixed float, sklearn thresholds by
    QUANTILE -- exactly 5% or 10% of each cluster is removed whether or
    not anomalies exist. Only "auto" lets the detector decide how many.
    """
    inlier = np.ones(len(embeddings), dtype=bool)

    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < MIN_CLUSTER_SIZE:
            continue
        X = embeddings[idx]

        if method == "isolation_forest":
            clf = IsolationForest(contamination=contamination,
                                  random_state=RANDOM_STATE)
            preds = clf.fit_predict(X)
        elif method == "lof":
            clf = LocalOutlierFactor(contamination=contamination, novelty=False)
            preds = clf.fit_predict(X)
        elif method == "random":
            if contamination == "auto":
                return None
            n_out = int(round(float(contamination) * len(idx)))
            preds = np.ones(len(idx), dtype=int)
            if n_out > 0:
                preds[rng.choice(len(idx), size=n_out, replace=False)] = -1
        else:
            raise ValueError(method)

        inlier[idx[preds == -1]] = False

    return inlier


# ---------------------------------------------------------------------------
# GRID BUILDING + GT SCORING
# ---------------------------------------------------------------------------
def build_grids(ctx, slide_of, rows, cols, prediction, inlier):
    grids = {}
    for slide_id, meta in ctx.items():
        sel = slide_of == slide_id
        before = np.zeros(meta["shape"], dtype=np.int32)
        after = np.zeros(meta["shape"], dtype=np.int32)
        r, c, g, keep = rows[sel], cols[sel], prediction[sel], inlier[sel]
        before[r, c] = g
        after[r, c] = np.where(keep, g, 0)
        grids[slide_id] = (before, after)
    return grids


def score_vs_gt(scar_prediction, meta):
    region = meta["region_mask"]
    algo = (scar_prediction > 0) & region
    gt = meta["gt_mask"] & region
    tp = int(np.sum(algo & gt))
    fp = int(np.sum(algo & ~gt))
    fn = int(np.sum(~algo & gt))
    tn = int(np.sum(~algo & ~gt & region))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def metrics_from_counts(c):
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    total = tp + fp + fn + tn
    def sd(a, b):
        return a / b if b else float("nan")
    return {
        "sensitivity": sd(tp, tp + fn),
        "specificity": sd(tn, tn + fp),
        "precision": sd(tp, tp + fp),
        "IoU": sd(tp, tp + fp + fn),
        "Dice": sd(2 * tp, 2 * tp + fp + fn),
        "accuracy": sd(tp + tn, total),
    }


# ---------------------------------------------------------------------------
# SPATIAL STRUCTURE OF THE REMOVED TILES
# ---------------------------------------------------------------------------
STRUCT8 = np.ones((3, 3), dtype=bool)


def contiguity_metrics(before, after, meta):
    """Restricted to the scored region, so these numbers match the GT
    metrics reported alongside them."""
    region = meta["region_mask"]
    removed = (before > 0) & (after == 0) & region
    kept = (after > 0) & region
    n = int(removed.sum())
    if n == 0:
        return None

    lab, ncomp = ndimage.label(removed, structure=STRUCT8)
    sizes = ndimage.sum(removed, lab, index=np.arange(1, ncomp + 1))
    largest = int(sizes.max()) if ncomp else 0

    k = np.ones((3, 3), dtype=np.float32)
    k[1, 1] = 0
    scar_before = (before > 0) & region
    nbr = ndimage.convolve(scar_before.astype(np.float32), k, mode="constant") / 8.0

    rem_nbr = ndimage.convolve(removed.astype(np.float32), k, mode="constant")
    singleton = removed & (rem_nbr == 0)

    gt = meta["gt_mask"] & region
    return {
        "n_removed": n,
        "n_components": int(ncomp),
        "largest_component": largest,
        "largest_frac": largest / n,
        "mean_component_size": float(sizes.mean()) if ncomp else 0.0,
        "median_component_size": float(np.median(sizes)) if ncomp else 0.0,
        "singleton_frac": float(singleton.sum()) / n,
        "frac_inside_gt": float(gt[removed].mean()),
        "nbr_frac_removed": float(nbr[removed].mean()),
        "nbr_frac_kept": float(nbr[kept].mean()) if kept.any() else float("nan"),
    }


def pool_contiguity(per_slide):
    """Count-weighted -- averaging per-slide fractions would let a slide
    with 40 removed tiles outvote one with 4,000."""
    rows = [r for r in per_slide if r]
    if not rows:
        return None
    n_tot = sum(r["n_removed"] for r in rows)
    out = {
        "n_removed": n_tot,
        "n_components": sum(r["n_components"] for r in rows),
        "largest_component": max(r["largest_component"] for r in rows),
    }
    out["largest_frac"] = out["largest_component"] / n_tot
    for key in ("singleton_frac", "frac_inside_gt", "nbr_frac_removed", "nbr_frac_kept"):
        vals = np.array([r[key] for r in rows], dtype=float)
        wts = np.array([r["n_removed"] for r in rows], dtype=float)
        ok = ~np.isnan(vals)
        out[key] = float(np.average(vals[ok], weights=wts[ok])) if ok.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
def clean_tissue_mask(tissue, open_iter=TISSUE_OPEN_ITER,
                      min_component=TISSUE_MIN_COMPONENT):
    """Removes thin streaks and small isolated blocks from the tissue mask.

    head_arr.sum(-1) != 0 flags any tile the model produced output for,
    which includes single-tile-wide lines along scan boundaries and small
    label/edge blocks. Filled with the tissue colour these render as thin
    stair-stepped outlines around and above the section, which a reader
    reads as deliberate annotation. Opening snaps the thin structures off,
    small components are dropped, and the surviving cores are regrown
    through the original mask so genuine tissue detail is preserved.
    """
    opened = ndimage.binary_opening(tissue, structure=STRUCT8, iterations=open_iter)
    lab, n = ndimage.label(opened, structure=STRUCT8)
    if n == 0:
        return tissue
    sizes = ndimage.sum(opened, lab, index=np.arange(1, n + 1))
    keep_labels = np.where(sizes >= min_component)[0] + 1
    if len(keep_labels) == 0:
        return tissue
    core = np.isin(lab, keep_labels)
    return ndimage.binary_dilation(core, structure=STRUCT8,
                                   iterations=open_iter + 1, mask=tissue)


def _bbox(mask, pad, shape):
    r, c = np.where(mask)
    if len(r) == 0:
        return None
    return (max(0, r.min() - pad), min(shape[0], r.max() + 1 + pad),
            max(0, c.min() - pad), min(shape[1], c.max() + 1 + pad))


def _draw(ax, disp, cmap):
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def removal_figure(slide_id, before, after, meta, arm_label, out_dir):
    """Two-panel publication figure.

    A: the whole section, with the scored region outlined, giving spatial
       context -- where the annotation sits within the tissue.
    B: the scored region alone, at the same native tile resolution but
       filling the panel, so individual excluded tiles are actually
       resolvable in print. At whole-section scale a single excluded tile
       is roughly one pixel and disappears; the detail panel is what makes
       the scattered pattern legible rather than a smudge.

    Both panels are drawn at native grid resolution with nearest-neighbour
    interpolation, so no excluded tile is lost to resampling.
    """
    region = meta["region_mask"]
    tissue = clean_tissue_mask(meta["tissue_mask"])

    removed_all = (before > 0) & (after == 0)
    kept_all = after > 0

    # 0 background, 1 tissue, 2 retained, 3 excluded
    disp = np.zeros(before.shape, dtype=np.uint8)
    disp[tissue] = 1
    disp[kept_all] = 2
    disp[removed_all] = 3
    cmap = ListedColormap([BACKGROUND_COLOUR, TISSUE_COLOUR,
                           KEPT_COLOUR, REMOVED_COLOUR])

    full = _bbox(tissue, CROP_PAD_TILES, before.shape)
    detail = _bbox(region, DETAIL_PAD_TILES, before.shape)
    if full is None or detail is None:
        return

    fr0, fr1, fc0, fc1 = full
    dr0, dr1, dc0, dc1 = detail

    disp_full = disp[fr0:fr1, fc0:fc1]
    disp_det = disp[dr0:dr1, dc0:dc1]
    gt_det = meta["gt_mask"][dr0:dr1, dc0:dc1].astype(float)
    region_det = region[dr0:dr1, dc0:dc1].astype(float)

    # size panels so both render at the same physical height
    aspect_full = disp_full.shape[1] / disp_full.shape[0]
    aspect_det = disp_det.shape[1] / disp_det.shape[0]
    w_full = PANEL_HEIGHT_IN * aspect_full
    w_det = PANEL_HEIGHT_IN * aspect_det

    fig = plt.figure(figsize=(w_full + w_det + 1.2, PANEL_HEIGHT_IN + 2.0),
                     facecolor=BACKGROUND_COLOUR)
    gs = fig.add_gridspec(1, 2, width_ratios=[aspect_full, aspect_det],
                          wspace=0.06, left=0.02, right=0.98,
                          top=0.88, bottom=0.10)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    _draw(axA, disp_full, cmap)
    # scored region marked on A as a dashed box, so panel B's extent is obvious
    axA.add_patch(Rectangle((dc0 - fc0 - 0.5, dr0 - fr0 - 0.5),
                            dc1 - dc0, dr1 - dr0,
                            fill=False, edgecolor=REGION_LINE_COLOUR,
                            linewidth=1.8, linestyle="--"))

    _draw(axB, disp_det, cmap)
    axB.contour(gt_det, levels=[0.5], colors=GT_LINE_COLOUR, linewidths=1.6)
    axB.contour(region_det, levels=[0.5], colors=REGION_LINE_COLOUR,
                linewidths=1.6, linestyles="dashed")

    axA.set_title("A   Whole section", fontsize=LABEL_FONTSIZE,
                  loc="left", pad=8)
    axB.set_title("B   Scored region (detail)", fontsize=LABEL_FONTSIZE,
                  loc="left", pad=8)

    handles = [
        Patch(facecolor=TISSUE_COLOUR, label="Tissue"),
        Patch(facecolor=KEPT_COLOUR, label="Retained scar prediction"),
        Patch(facecolor=REMOVED_COLOUR, label="Excluded as outlier"),
        Line2D([0], [0], color=GT_LINE_COLOUR, lw=1.6,
               label="Ground-truth scar"),
        Line2D([0], [0], color=REGION_LINE_COLOUR, lw=1.6, ls="--",
               label="Scored region"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 0.005))

    stats = contiguity_metrics(before, after, meta)
    header = slide_id
    if stats:
        header += (f"    excluded n = {stats['n_removed']:,} in "
                   f"{stats['n_components']:,} separate fragments; "
                   f"largest {stats['largest_frac']*100:.1f}%, "
                   f"isolated {stats['singleton_frac']*100:.1f}%, "
                   f"inside GT {stats['frac_inside_gt']*100:.1f}%")
    fig.suptitle(header, fontsize=SUPTITLE_FONTSIZE, y=0.975)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{slide_id}_removal"
    fig.savefig(stem.with_suffix(".png"), dpi=FIG_DPI,
                facecolor=BACKGROUND_COLOUR, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),
                facecolor=BACKGROUND_COLOUR, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def run_arm(level, method, contamination, embeddings, labels, prediction,
            ctx, slide_of, rows, cols, rng, silhouette_sample_idx):
    inlier = flag_outliers(embeddings, labels, method, contamination, rng)
    if inlier is None:
        return None, None

    grids = build_grids(ctx, slide_of, rows, cols, prediction, inlier)

    pooled_before = defaultdict(int)
    pooled_after = defaultdict(int)
    contig_rows, per_slide_rows = [], []

    for slide_id, (before, after) in grids.items():
        meta = ctx[slide_id]
        cb = score_vs_gt(before, meta)
        ca = score_vs_gt(after, meta)
        for k in ("TP", "FP", "FN", "TN"):
            pooled_before[k] += cb[k]
            pooled_after[k] += ca[k]

        cg = contiguity_metrics(before, after, meta)
        contig_rows.append(cg)

        mb, ma = metrics_from_counts(cb), metrics_from_counts(ca)
        row = {"level": level, "method": method, "contamination": contamination,
               "slide_id": slide_id,
               "removed_TP": cb["TP"] - ca["TP"], "removed_FP": cb["FP"] - ca["FP"]}
        for m in ("sensitivity", "specificity", "precision", "IoU", "Dice"):
            row[f"{m}_before"] = mb[m]
            row[f"{m}_after"] = ma[m]
        if cg:
            row.update({f"contig_{k}": v for k, v in cg.items()})
        per_slide_rows.append(row)

    mb, ma = metrics_from_counts(pooled_before), metrics_from_counts(pooled_after)
    pooled_contig = pool_contiguity(contig_rows)

    sil = float("nan")
    keep_idx = silhouette_sample_idx[inlier[silhouette_sample_idx]]
    if len(np.unique(labels[keep_idx])) >= 2:
        sil = float(silhouette_score(embeddings[keep_idx], labels[keep_idx]))

    n_removed = int((~inlier).sum())
    summary = {
        "level": level, "method": method, "contamination": contamination,
        "n_removed": n_removed,
        "pct_removed": 100 * n_removed / len(inlier),
        "silhouette_after": sil,
        "removed_TP": pooled_before["TP"] - pooled_after["TP"],
        "removed_FP": pooled_before["FP"] - pooled_after["FP"],
    }
    tp_lost = summary["removed_TP"]
    fp_lost = summary["removed_FP"]
    summary["TP_lost_per_FP_removed"] = tp_lost / fp_lost if fp_lost else float("nan")
    for m in ("sensitivity", "specificity", "precision", "IoU", "Dice"):
        summary[f"{m}_before"] = mb[m]
        summary[f"{m}_after"] = ma[m]
        summary[f"{m}_delta"] = ma[m] - mb[m]
    if pooled_contig:
        summary.update({f"contig_{k}": v for k, v in pooled_contig.items()})

    return summary, (per_slide_rows, grids)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'")
    print(f"Output folder: {OUTPUT_DIR}\n")

    print("Loading slide context (tissue / valid / GT masks)...")
    ctx = load_slide_context()
    if not ctx:
        print("No slides available -- check paths.")
        return

    print("\nPooling scar-candidate embeddings (keeping tile coordinates)...")
    embeddings, slide_of, rows, cols = pool_scar_candidates(ctx)
    print(f"  pooled: {len(embeddings):,} tiles")

    rng = np.random.default_rng(RANDOM_STATE)
    n_sil = min(SUBSAMPLE_N, len(embeddings))
    silhouette_sample_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(embeddings), size=n_sil, replace=False)

    all_summary, all_per_slide = [], []

    for level, centres_path in (("sub", SCAR_CENTRES_5_PATH),
                                ("macro", SCAR_CENTRES_4_PATH)):
        if not centres_path.exists():
            print(f"\n{level}: SKIPPED - {centres_path.name} not found")
            continue
        print(f"\n{'='*72}\n{level.upper()} LEVEL\n{'='*72}")
        centres = np.load(centres_path)
        labels = assign_clusters(embeddings, centres)
        prediction = baseline_scar_prediction(level, labels)
        print(f"  baseline scar tiles: {int((prediction > 0).sum()):,} "
              f"of {len(prediction):,} candidates")

        base_sil = float(silhouette_score(embeddings[silhouette_sample_idx],
                                          labels[silhouette_sample_idx]))
        print(f"  baseline silhouette: {base_sil:.3f}")

        for method in METHODS:
            for contamination in CONTAMINATION_VALUES:
                summary, extra = run_arm(
                    level, method, contamination, embeddings, labels, prediction,
                    ctx, slide_of, rows, cols, rng, silhouette_sample_idx)
                if summary is None:
                    continue
                summary["silhouette_baseline"] = base_sil
                summary["silhouette_delta"] = summary["silhouette_after"] - base_sil
                all_summary.append(summary)
                all_per_slide.extend(extra[0])

                print(f"\n  [{method}, contamination={contamination}]  "
                      f"removed {summary['n_removed']:,} ({summary['pct_removed']:.1f}%)")
                print(f"    sensitivity {summary['sensitivity_before']:.3f} -> "
                      f"{summary['sensitivity_after']:.3f}   "
                      f"IoU {summary['IoU_before']:.3f} -> {summary['IoU_after']:.3f}")
                print(f"    TP lost {summary['removed_TP']:,}  "
                      f"FP removed {summary['removed_FP']:,}  "
                      f"ratio {summary['TP_lost_per_FP_removed']:.2f} TP per FP")
                if "contig_largest_frac" in summary:
                    print(f"    SPATIAL: largest component = "
                          f"{summary['contig_largest_frac']*100:.1f}% of removals   "
                          f"isolated = {summary['contig_singleton_frac']*100:.1f}%   "
                          f"inside GT = {summary['contig_frac_inside_gt']*100:.1f}%")
                    print(f"             scar-neighbour fraction: "
                          f"removed={summary['contig_nbr_frac_removed']:.2f}  "
                          f"kept={summary['contig_nbr_frac_kept']:.2f}")

                if (level, method, contamination) in FIGURE_ARMS:
                    arm_dir = OUTPUT_DIR / f"figures_{level}_{method}_{contamination}"
                    label = f"{level}, {method}, contamination={contamination}"
                    for slide_id, (before, after) in extra[1].items():
                        removal_figure(slide_id, before, after, ctx[slide_id],
                                       label, arm_dir)
                    print(f"    figures -> {arm_dir.name}/")

    if not all_summary:
        print("\nNo arms completed.")
        return

    sdf = pd.DataFrame(all_summary)
    pdf = pd.DataFrame(all_per_slide)
    sdf.to_csv(OUTPUT_DIR / "arm_summary_pooled.csv", index=False)
    pdf.to_csv(OUTPUT_DIR / "arm_summary_per_slide.csv", index=False)

    print(f"\n{'='*72}\nSPATIAL STRUCTURE OF REMOVED TILES (pooled)\n{'='*72}")
    cols_show = ["level", "method", "contamination", "n_removed",
                 "contig_largest_frac", "contig_singleton_frac",
                 "contig_frac_inside_gt", "TP_lost_per_FP_removed"]
    have = [c for c in cols_show if c in sdf.columns]
    print(sdf[have].to_string(index=False))

    print(f"\nSaved: {OUTPUT_DIR / 'arm_summary_pooled.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'arm_summary_per_slide.csv'}")
    print(
        "\nHOW TO READ contig_largest_frac / contig_singleton_frac:\n"
        "  A 'dense, contiguous core' means largest_frac is high (say >0.5)\n"
        "  and singleton_frac is low. Scattered noise is the opposite: many\n"
        "  small components, largest_frac near 0, singleton_frac high.\n"
        "  Note that removed tiles can be BOTH inside the GT scar region\n"
        "  (high frac_inside_gt) AND scattered (low largest_frac); those are\n"
        "  independent claims and should be reported separately.\n"
        "  Compare each detector against the RANDOM arm at the same\n"
        "  contamination, not against the baseline: removing points inflates\n"
        "  silhouette mechanically, so random removal is the real control."
    )


if __name__ == "__main__":
    main()
