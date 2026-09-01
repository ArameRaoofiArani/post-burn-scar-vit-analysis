# python figure_score_distributions.py
#
# Distribution of the raw scar-channel score, split by outcome
# (TP / FP / FN / TN), pooled across slides and per slide.
#
# Produces:
#   score_distribution_pooled.png/.pdf    KDE curves, all slides pooled
#   score_distribution_by_slide.png/.pdf  2 x 3 plate, panels a) to f)
#
# The outcome split matches 03_extract_ground_truth.py exactly:
#   region_mask = valid_mask & tissue_mask
#   algo_scar   = (scar_prediction > 0) & region_mask
#   gt_scar     = gt_mask & region_mask
#
# The prediction is rebuilt from the embeddings and scar_centres_5 rather
# than read from *_scar_prediction.npy, so a modified grid on disk cannot affect
# the figure.
#
# Each curve is normalised to integrate to 1, so the plot compares the
# SHAPE of the four distributions rather than their sizes; group sizes are
# in the legend. Without that, TN and TP would swamp FP entirely.
#
# Writes into its own timestamped folder.

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity

# ---------------------------------------------------------------------------
import config                           # noqa: E402
from config import (                    # noqa: E402
    MODEL_OUTPUT_DIR,
    OUTPUT_DIR as FIGURE_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES as SLIDE_IDS,
    VALID_CLUSTER_INDICES,
)

OUTPUT_DIR = FIGURE_DIR / f"score_distribution_{datetime.now():%Y%m%d_%H%M%S}"

# ---- style ----------------------------------------------------------------
GROUP_COLOURS = {
    "TP": "green",
    "FP": "red",
    "FN": "blue",
    "TN": "gray",
}
GROUP_ORDER = ["TP", "FP", "FN", "TN"]
GROUP_LABELS = {"TP": "TP", "FP": "FP", "FN": "FN", "TN": "TN"}

X_LABEL = "Raw scar-channel score"
Y_LABEL = "Density"

LINE_WIDTH = 2.8
N_GRID = 400
MAX_KDE_SAMPLES = 20000    # KernelDensity is O(n_samples x n_grid), so
                           # large groups are subsampled with a fixed seed.
                           # The curve is visually identical and it finishes
                           # in seconds rather than minutes.
RNG_SEED = 0

AXIS_LABEL_FS = 28
TICK_FS = 22
LEGEND_FS = 22
PANEL_TITLE_FS = 23
LABEL_PAD = 14             # raise this rather than shrinking the font if
                           # the axis label crowds the tick numbers
SAVE_DPI = 600
SAVE_PDF = True

FULL_BOX = True            # box on all four sides

PANEL_LETTERS = "abcdef"
NCOLS, NROWS = 3, 2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.labelsize": AXIS_LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "axes.spines.top": FULL_BOX,
    "axes.spines.right": FULL_BOX,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
def load_slide_groups(slide_id, centres):
    """Rebuilds the baseline prediction, then splits the raw scar-channel
    score by outcome within the scored region."""
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_head_arr.npy")
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_id}_scar_norm_arr.npy")
    gt_mask = np.load(config.gt_mask_path(slide_id)).astype(bool)
    valid_mask = np.load(config.valid_mask_path(slide_id)).astype(bool)

    tissue_mask = head_arr.sum(axis=-1) != 0
    region_mask = valid_mask & tissue_mask

    predicted_class = np.argmax(head_arr, axis=2)
    is_candidate = (predicted_class == SCAR_CLASS_INDEX) & tissue_mask

    prediction = np.zeros(head_arr.shape[:2], dtype=np.int32)
    r, c = np.where(is_candidate)
    if len(r):
        nearest = np.argmin(cdist(norm_arr[r, c, :], centres), axis=1)
        keep = np.isin(nearest, list(VALID_CLUSTER_INDICES))
        prediction[r[keep], c[keep]] = nearest[keep] + 1

    algo_scar = (prediction > 0) & region_mask
    gt_scar = gt_mask & region_mask
    score = head_arr[..., SCAR_CLASS_INDEX]

    return {
        "TP": score[algo_scar & gt_scar],
        "FP": score[algo_scar & ~gt_scar],
        "FN": score[~algo_scar & gt_scar],
        "TN": score[~algo_scar & ~gt_scar & region_mask],
    }


def silverman_bandwidth(scores):
    """One shared bandwidth for every group. Scaling the bandwidth to each
    group's own range instead would smooth the curves by different amounts,
    so apparent width differences between TP and FP would partly be an
    artefact of the smoothing rather than the data."""
    n = len(scores)
    if n < 2:
        return 1e-3
    std = float(np.std(scores))
    iqr = float(np.subtract(*np.percentile(scores, [75, 25])))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = std if std > 0 else 1e-3
    return max(0.9 * sigma * n ** (-1 / 5), 1e-4)


def kde_curve(scores, grid, bandwidth, rng):
    if len(scores) > MAX_KDE_SAMPLES:
        scores = rng.choice(scores, MAX_KDE_SAMPLES, replace=False)
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(np.asarray(scores).reshape(-1, 1))
    return np.exp(kde.score_samples(grid.reshape(-1, 1)))


def make_grid(all_scores, n=N_GRID, pad_frac=0.05):
    """Evaluation grid spanning the data plus a small RELATIVE margin. A
    fixed margin would be enormous on a 0-1 score and negligible on a
    0-100 one."""
    lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
    pad = max((hi - lo) * pad_frac, 1e-6)
    return np.linspace(lo - pad, hi + pad, n)


def draw_distribution(ax, groups, grid, bandwidth, rng, show_legend=True,
                      show_counts=True):
    present = [(k, groups[k]) for k in GROUP_ORDER if len(groups.get(k, [])) >= 5]
    if not present:
        return False

    for name, scores in present:
        label = GROUP_LABELS[name]
        if show_counts:
            label += f" (n={len(scores):,})"
        ax.plot(grid, kde_curve(scores, grid, bandwidth, rng),
                color=GROUP_COLOURS[name], linewidth=LINE_WIDTH, label=label)

    ax.set_ylim(bottom=0)
    # limits from the GRID, not the data range: the grid is padded, so
    # clamping to the data would clip the TP curve mid-peak at the edge
    ax.set_xlim(grid[0], grid[-1])
    if show_legend:
        ax.legend(frameon=True, loc="upper center")
    return True


# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    centres = np.load(SCAR_CENTRES_PATH)
    rng = np.random.default_rng(RNG_SEED)
    print(f"Output folder: {OUTPUT_DIR}\n")

    per_slide, pooled = {}, {k: [] for k in GROUP_ORDER}
    for slide_id in SLIDE_IDS:
        try:
            groups = load_slide_groups(slide_id, centres)
        except FileNotFoundError as e:
            print(f"  {slide_id}: SKIPPED - {e}")
            continue
        per_slide[slide_id] = groups
        for k in GROUP_ORDER:
            pooled[k].append(groups[k])
        print(f"{slide_id}")
        for k in GROUP_ORDER:
            v = groups[k]
            mean_v = float(v.mean()) if len(v) else float("nan")
            print(f"    {k}: n = {len(v):>8,}   mean = {mean_v:.3f}")

    if not per_slide:
        print("No slides loaded.")
        return

    pooled = {k: (np.concatenate(v) if v else np.array([]))
              for k, v in pooled.items()}

    print("\nPOOLED")
    for k in GROUP_ORDER:
        v = pooled[k]
        mean_v = float(v.mean()) if len(v) else float("nan")
        print(f"    {k}: n = {len(v):>8,}   mean = {mean_v:.3f}")

    # one shared grid and one shared bandwidth, so every panel sits on the
    # same x-axis and is smoothed by the same amount
    all_scores = np.concatenate([v for v in pooled.values() if len(v)])
    grid = make_grid(all_scores)
    bandwidth = silverman_bandwidth(all_scores)

    # ---- pooled figure ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7.5), layout="constrained")
    draw_distribution(ax, pooled, grid, bandwidth, rng)
    ax.set_xlabel(X_LABEL, labelpad=LABEL_PAD)
    ax.set_ylabel(Y_LABEL, labelpad=LABEL_PAD)

    stem = OUTPUT_DIR / "score_distribution_pooled"
    fig.savefig(stem.with_suffix(".png"), dpi=SAVE_DPI, facecolor="white")
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {stem.name}.png / .pdf")

    # ---- per-slide plate --------------------------------------------------
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(7.0 * NCOLS, 5.4 * NROWS),
                             sharex=True, layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for i, (slide_id, groups) in enumerate(per_slide.items()):
        ax = axes[i]
        draw_distribution(ax, groups, grid, bandwidth, rng,
                          show_legend=False, show_counts=False)
        ax.set_title(f"{PANEL_LETTERS[i]})", fontsize=PANEL_TITLE_FS,
                     loc="left", pad=6)
        if i % NCOLS == 0:
            ax.set_ylabel(Y_LABEL, labelpad=LABEL_PAD)
        if i >= len(per_slide) - NCOLS:
            ax.set_xlabel(X_LABEL, labelpad=LABEL_PAD)

    for j in range(len(per_slide), len(axes)):
        axes[j].axis("off")

    handles = [plt.Line2D([0], [0], color=GROUP_COLOURS[k], lw=LINE_WIDTH)
               for k in GROUP_ORDER]
    labels = [GROUP_LABELS[k] for k in GROUP_ORDER]
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False, fontsize=LEGEND_FS)

    stem = OUTPUT_DIR / "score_distribution_by_slide"
    fig.savefig(stem.with_suffix(".png"), dpi=SAVE_DPI, facecolor="white")
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Saved: {stem.name}.png / .pdf")

    print("\nPanel order for the caption:")
    for i, slide_id in enumerate(per_slide):
        print(f"  {PANEL_LETTERS[i]}) {slide_id}")


if __name__ == "__main__":
    main()