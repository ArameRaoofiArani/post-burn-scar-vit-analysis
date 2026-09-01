"""Shared scoring code.

Every script that compares a scar prediction to the ground truth needs the same
things: load the grid and masks, restrict to the scored region, count
TP/FP/FN/TN, and turn those counts into metrics. That lives here so the
numbers cannot drift between scripts.

Two conventions used throughout:

* **positive = scar.** Some scripts store labels the other way round
  (0 = scar, 1 = rest); they convert to booleans before calling in here.
* **the scored region** is `valid_mask & tissue_mask` — the annotated
  tissue piece, minus any tile the model wrote no output for. Every count
  below is taken inside it, so un-annotated tissue never enters a metric.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_curve, auc

import config
from config import MODEL_OUTPUT_DIR, SCAR_CLASS_INDEX


def safe_div(num, den):
    """Division that returns NaN instead of raising when the denominator is
    zero -- a metric with no support is undefined, not an error."""
    return num / den if den > 0 else float("nan")


def load_scored_region(slide_id, scar_prediction_path):
    """Load one slide and return (algo_scar, gt_mask, region_mask, scar_score).

    The shape check is deliberate: a grid built from a different tile
    grid would otherwise broadcast silently and produce meaningless counts.
    """
    scar_prediction = np.load(scar_prediction_path)
    gt_mask = np.load(config.gt_mask_path(slide_id)).astype(bool)
    valid_mask = np.load(config.valid_mask_path(slide_id)).astype(bool)
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_head_arr.npy")

    if scar_prediction.shape != gt_mask.shape or scar_prediction.shape != valid_mask.shape:
        raise ValueError(
            f"{slide_id}: shape mismatch -- scar_prediction {scar_prediction.shape}, "
            f"gt_mask {gt_mask.shape}, valid_mask {valid_mask.shape}"
        )

    tissue_mask = head_arr.sum(axis=-1) != 0
    region_mask = valid_mask & tissue_mask
    algo_scar = scar_prediction > 0
    scar_score = head_arr[..., SCAR_CLASS_INDEX]
    return algo_scar, gt_mask, region_mask, scar_score


def binary_counts(algo_scar, gt_mask, region_mask):
    """TP/FP/FN/TN inside the scored region, positive = scar."""
    a = algo_scar & region_mask
    g = gt_mask & region_mask

    tp = int(np.sum(a & g))
    fp = int(np.sum(a & ~g))
    fn = int(np.sum(~a & g))
    tn = int(np.sum(~a & ~g & region_mask))
    return tp, fp, fn, tn


def rate_metrics(tp, fp, fn, tn):
    """The six rate metrics, short key names. Single source of the arithmetic."""
    total = tp + fp + fn + tn
    return {
        "sensitivity": safe_div(tp, tp + fn),          # a.k.a. recall
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),            # a.k.a. PPV
        "IoU": safe_div(tp, tp + fp + fn),
        "Dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "accuracy": safe_div(tp + tn, total),
    }


def metrics_from_counts(tp, fp, fn, tn):
    """Counts plus rate metrics, using the longer key names the per-slide
    metrics tables report."""
    r = rate_metrics(tp, fp, fn, tn)
    return {
        "n_tiles_compared": tp + fp + fn + tn,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": r["accuracy"],
        "sensitivity_recall": r["sensitivity"],
        "specificity": r["specificity"],
        "precision_ppv": r["precision"],
        "IoU": r["IoU"],
        "Dice": r["Dice"],
    }


def sklearn_metrics(algo_scar, gt_mask, region_mask):
    """Balanced accuracy and F1 on the flattened, region-masked masks.

    Positive = scar, matching everything else here, so no pos_label is
    needed. Returns NaN when only one class is present -- those scores
    aren't defined, and sklearn would warn or error.
    """
    y_true = gt_mask[region_mask].astype(int)
    y_pred = algo_scar[region_mask].astype(int)

    if len(np.unique(y_true)) < 2:
        return {"balanced_accuracy": float("nan"), "f1_score": float("nan")}

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def pool_counts(counts_list):
    """Sum TP/FP/FN/TN across slides. Pooling counts and then computing rates
    is not the same as averaging per-slide rates -- it weights each slide by
    how many tiles it contributes, which is what the pooled row reports."""
    out = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for c in counts_list:
        for k in out:
            out[k] += c[k]
    return out


def score_grid(slide_id, scar_prediction_path):
    """Counts and rate metrics for one scar prediction, in one call."""
    algo_scar, gt_mask, region_mask, _ = load_scored_region(slide_id, scar_prediction_path)
    tp, fp, fn, tn = binary_counts(algo_scar, gt_mask, region_mask)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, **rate_metrics(tp, fp, fn, tn)}


def percentile_rank(x):
    """0..1 percentile rank of each element within x. 1.0 = highest.

    The decision threshold is defined against this ranking, so every script
    that applies or calibrates it must use the identical implementation.
    Ranking within a slide rather than thresholding the raw score directly
    is deliberate: absolute score ranges differ substantially across slides.
    """
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    if len(x) > 1:
        ranks /= (len(x) - 1)
    return ranks


def nearest_cluster(embeddings, centres, chunk=20000):
    """Index of the nearest centroid for each embedding, by Euclidean
    distance. Chunked because the full pairwise broadcast over a whole
    slide's tiles is large enough to matter.
    """
    n = embeddings.shape[0]
    out = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = np.linalg.norm(embeddings[start:end][:, None, :] - centres[None, :, :], axis=-1)
        out[start:end] = np.argmin(d, axis=1)
    return out


def load_outcome_masks(slide_id, scar_prediction_path):
    """(gt, algo, fn) masks for the false-negative characterisation.

    NOTE the region differs from load_scored_region(): here it is the valid
    mask ALONE, without also requiring the tile to contain tissue. That is
    the convention the false-negative analysis was run under, so it is kept.
    A ground-truth tile the model wrote no output for counts as a false
    negative here but is excluded from 04_evaluate_agreement.py's counts, so FN
    totals from the two can differ slightly.
    """
    gt_full = np.load(config.gt_mask_path(slide_id))
    valid = np.load(config.valid_mask_path(slide_id)).astype(bool)
    scar_prediction = np.load(scar_prediction_path)

    gt = (gt_full > 0) & valid
    algo = (scar_prediction > 0) & valid
    return gt, algo, gt & ~algo


# ---------------------------------------------------------------- figures
def plot_roc(y_true, y_score, label, out_path):
    """ROC of the raw scar-channel score against the ground truth.

    Note this scores the raw score, not the thresholded prediction, so the
    AUC is a property of the score itself and does not move when the
    classification rule changes.
    """
    if len(np.unique(y_true)) < 2:
        print(f"  Skipping ROC for {label} -- only one class present in region.")
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{label}\nROC: raw scar-class score vs GT scar/not-scar")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_confusion(tp, fp, fn, tn, title, out_path,
                   pred_labels=("Pred: not scar", "Pred: scar"),
                   gt_labels=("GT: not scar", "GT: scar"),
                   figsize=(4, 4)):
    """2x2 confusion matrix, laid out [[TN, FP], [FN, TP]].

    The tick labels are arguments because scripts using the inverted
    0=scar/1=rest convention need to say so on the axes.
    """
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(matrix, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{matrix[i, j]}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(pred_labels))
    ax.set_yticks([0, 1]); ax.set_yticklabels(list(gt_labels))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
