"""
Diagnostic -- how separable are the clusters in embedding space?

    python analysis_cluster_separation.py

Pools scar-candidate embeddings across slides, then reports silhouette
scores and PCA scatter panels for three labellings of the same points:
the four macro clusters, the five scar sub-clusters, and the ground truth.
Nothing downstream depends on its outputs.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

import config
from config import (
    MODEL_OUTPUT_DIR,
    GT_DIR,                             # {slide}_gt_scar_mask.npy lives here
    SCAR_CENTRES_4_PATH,
    SCAR_CENTRES_5_PATH,
    SCAR_CLASS_INDEX,                   # head_arr channel gating scar candidates
    SLIDE_NAMES,
    HEAD_CLASS_NAMES as MACRO_LABELS,
)

SUBSAMPLE_SIZE = 6000   # per clustering, randomly sampled from pooled scar-candidate tiles
RANDOM_STATE = 0

OUTPUT_DIR = config.OUTPUT_DIR / "cluster_separation"
GT_LABEL_NAMES = {0: "non-scar (GT)", 1: "scar (GT)"}

# --- Publication figure settings -------------------------------------------
# Okabe-Ito palette: colorblind-safe, standard for scientific figures.
# Index order chosen so category colors stay visually distinct and consistent
# across all three panels (macro / sub / GT all draw from this same list).
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Ground truth uses its OWN color pair, deliberately different from the
# categorical cluster palette above. Panels A/B color arbitrary cluster
# identities (which cluster does a tile fall into); panel C colors a
# biological positive/negative label (is this tile actually scar). Reusing
# the same blue/orange for both implies an equivalence that isn't there
# (e.g. "cluster 1 = GT scar") -- a gray/red negative-positive convention
# keeps the two kinds of label visually and conceptually distinct.
GT_COLORS = {0: "#999999", 1: "#B2182B"}  # gray = non-scar, red = scar

FIG_DPI = 300
MARKER_SIZE = 9         # up from 6: the larger panels below give points more
MARKER_ALPHA = 0.45     # room, and small dots read as noise at 300 dpi in print

# ---- font sizes (all bumped for publication) ------------------------------
AXIS_LABEL_FONTSIZE = 22
TICK_FONTSIZE = 16
PANEL_TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
PANEL_LETTER_FONTSIZE = 26

# ---- panel geometry -------------------------------------------------------
PANEL_WIDTH = 7.5       # inches
PANEL_HEIGHT = 6.0      # inches, per panel in the stacked combined figure

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": TICK_FONTSIZE,
    "axes.titlesize": PANEL_TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,     # embed TrueType so text stays selectable/editable
    "ps.fonttype": 42,
})


def collect_scar_candidate_embeddings():
    """Pools embeddings for every scar-candidate tile (head_arr channel-2
    argmax), across all slides -- the same population the pipeline scores.

    Returns:
        all_embeddings: (N, D) float array of pooled norm embeddings
        all_coords: (N, 3) object array of (slide_name, row, col) for each
                    embedding, in the same order, so downstream code can
                    look up per-tile metadata (e.g. GT labels) later.
    """
    all_embeddings = []
    all_coords = []
    for slide_name in SLIDE_NAMES:
        head_path = MODEL_OUTPUT_DIR / f"{slide_name}_scar_head_arr.npy"
        norm_path = MODEL_OUTPUT_DIR / f"{slide_name}_scar_norm_arr.npy"
        if not (head_path.exists() and norm_path.exists()):
            print(f"  {slide_name}: SKIPPED - prediction files not found")
            continue
        head_arr = np.load(head_path)
        norm_arr = np.load(norm_path)
        has_tissue = np.any(head_arr != 0, axis=2)
        predicted_class = np.argmax(head_arr, axis=2)
        is_scar_candidate = (predicted_class == SCAR_CLASS_INDEX) & has_tissue
        idx = np.argwhere(is_scar_candidate)
        if len(idx) == 0:
            continue
        embeddings = norm_arr[idx[:, 0], idx[:, 1], :]
        all_embeddings.append(embeddings)
        # store (slide_name, row, col) per tile, matching embeddings order
        coords = np.empty((len(idx), 3), dtype=object)
        coords[:, 0] = slide_name
        coords[:, 1] = idx[:, 0]
        coords[:, 2] = idx[:, 1]
        all_coords.append(coords)
        print(f"  {slide_name}: {len(idx)} scar-candidate tiles")
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_coords, axis=0)


def lookup_gt_labels(coords, gt_dir=GT_DIR):
    """Maps each (slide_name, row, col) tile coordinate to its ground-truth
    scar/non-scar label, read from {slide}_gt_scar_mask.npy (bool, same
    shape as the tile grid).

    coords: (N, 3) array-like of (slide_name, row, col)
    returns: (N,) int array, 1 = GT scar, 0 = GT non-scar
    """
    gt_cache = {}
    labels = np.empty(len(coords), dtype=int)
    for i, (slide_name, row, col) in enumerate(coords):
        if slide_name not in gt_cache:
            gt_cache[slide_name] = np.load(gt_dir / f"{slide_name}_gt_scar_mask.npy")
        labels[i] = int(gt_cache[slide_name][int(row), int(col)])
    return labels


def nearest_cluster(embeddings, centres):
    dist = np.linalg.norm(embeddings[:, None, :] - centres[None, :, :], axis=2)
    return np.argmin(dist, axis=1)


def report_silhouette(name, embeddings, labels, label_names=None):
    unique = np.unique(labels)
    if len(unique) < 2:
        print(f"  [{name}] only {len(unique)} cluster present in sample -- skipping silhouette")
        return None
    score = silhouette_score(embeddings, labels)
    per_sample = silhouette_samples(embeddings, labels)
    print(f"\n  [{name}] overall silhouette score: {score:.4f}")
    print("    (1.0 = perfectly separated, 0 = boundary/ambiguous, <0 = likely mislabeled)")
    for c in unique:
        mean_s = per_sample[labels == c].mean()
        n = int(np.sum(labels == c))
        lname = label_names.get(c, f"cluster {c}") if label_names else f"cluster {c}"
        print(f"    {lname:<16} n={n:<6} mean silhouette={mean_s:.4f}")
    return score, per_sample


def compute_shared_pca(embeddings):
    """Fits PCA once on the sampled embeddings. Because the SAME sample is
    reused for macro/sub/GT coloring, this single fit gives all three panels
    identical PC1/PC2 coordinates and axis limits -- points land in the same
    place in every panel. Shared PCA coordinates allow direct comparison
    across panels."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords_2d = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    # shared axis limits with a small margin, reused by every panel
    pad_x = 0.05 * (coords_2d[:, 0].max() - coords_2d[:, 0].min())
    pad_y = 0.05 * (coords_2d[:, 1].max() - coords_2d[:, 1].min())
    xlim = (coords_2d[:, 0].min() - pad_x, coords_2d[:, 0].max() + pad_x)
    ylim = (coords_2d[:, 1].min() - pad_y, coords_2d[:, 1].max() + pad_y)
    return coords_2d, explained, xlim, ylim


def _draw_panel(ax, coords_2d, labels, xlim, ylim,
                 label_names=None, legend_loc="best", panel_letter=None,
                 title=None, colors=None, show_xlabel=True, show_ylabel=True):
    """Draws one PCA scatter panel onto an existing Axes. Shared by both the
    single-panel and combined-figure entry points so styling never drifts
    between the two.

    Axis labels are plain "PC1"/"PC2" -- the explained-variance percentages
    are printed to the console instead, so they can go in the figure caption
    where a reader expects them rather than crowding the axis at 22pt.

    colors: optional dict {label_value: hex_color}. If omitted, colors are
    assigned from OKABE_ITO by position -- use this default for arbitrary
    cluster identities (panels A/B). Pass an explicit dict (e.g. GT_COLORS)
    when the label carries its own meaning that shouldn't borrow the cluster
    palette (panel C)."""
    unique = np.unique(labels)
    for i, c in enumerate(unique):
        mask = labels == c
        lname = label_names.get(c, f"cluster {c}") if label_names else f"cluster {c}"
        color = colors[c] if colors is not None else OKABE_ITO[i % len(OKABE_ITO)]
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   s=MARKER_SIZE, alpha=MARKER_ALPHA, linewidths=0,
                   color=color,
                   label=f"{lname} (n = {int(mask.sum()):,})",
                   rasterized=True)  # rasterize points, keep axes/text vector
    if show_xlabel:
        ax.set_xlabel("PC1", labelpad=8)
    if show_ylabel:
        ax.set_ylabel("PC2", labelpad=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title:
        ax.set_title(title, pad=10)
    if panel_letter:
        ax.text(-0.10, 1.08, panel_letter, transform=ax.transAxes,
                 fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold",
                 va="top", ha="left")
    leg = ax.legend(loc=legend_loc, markerscale=4, frameon=False,
                     handletextpad=0.4, borderaxespad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(0.9)


def pca_plot(coords_2d, explained, labels, xlim, ylim, title, out_path_stem,
             label_names=None, panel_letter=None, colors=None):
    """Saves ONE publication-ready panel as both .png (300 dpi, for Word)
    and .pdf (vector, for print/LaTeX). out_path_stem should have NO
    extension -- both files are written from it."""
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT), layout="constrained")
    _draw_panel(ax, coords_2d, labels, xlim, ylim,
                label_names=label_names, panel_letter=panel_letter, title=title,
                colors=colors)
    png_path = Path(str(out_path_stem) + ".png")
    pdf_path = Path(str(out_path_stem) + ".pdf")
    fig.savefig(png_path, dpi=FIG_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path.name}, {pdf_path.name}")


def combined_pca_figure(coords_2d, explained, xlim, ylim,
                         macro_labels, sub_labels, gt_labels, out_path_stem):
    """Single 3-panel (A/B/C) figure combining macro/sub/GT coloring, stacked
    VERTICALLY (one panel under the next) and sharing PC axes/limits.

    Stacking rather than sitting side by side means each panel keeps a full
    column width, so the points and the enlarged fonts both have room; with
    three panels across, a journal's single-column width would shrink each
    scatter to the point where the cluster structure stops being readable.

    layout="constrained" is used instead of tight_layout: it reserves space
    for titles, panel letters and axis labels before placing the axes, so the
    larger fonts can't overlap the panel above.
    """
    fig, axes = plt.subplots(
        3, 1,
        figsize=(PANEL_WIDTH, PANEL_HEIGHT * 3),
        sharex=True, sharey=True,
        layout="constrained",
    )

    _draw_panel(axes[0], coords_2d, macro_labels, xlim, ylim,
                label_names=MACRO_LABELS, panel_letter="A",
                title="Macro cluster (scar_centres_4)", show_xlabel=False)
    _draw_panel(axes[1], coords_2d, sub_labels, xlim, ylim,
                label_names=None, panel_letter="B",
                title="Sub cluster (scar_centres_5)", show_xlabel=False)
    _draw_panel(axes[2], coords_2d, gt_labels, xlim, ylim,
                label_names=GT_LABEL_NAMES, panel_letter="C",
                title="Ground truth", colors=GT_COLORS, show_xlabel=True)

    png_path = Path(str(out_path_stem) + ".png")
    pdf_path = Path(str(out_path_stem) + ".pdf")
    fig.savefig(png_path, dpi=FIG_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved combined figure: {png_path.name}, {pdf_path.name}")


def main():
    config.ensure_dirs(OUTPUT_DIR)
    print("Collecting scar-candidate embeddings across all slides...")
    embeddings, coords = collect_scar_candidate_embeddings()
    print(f"\nTotal pooled scar-candidate tiles: {len(embeddings)}")

    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = min(SUBSAMPLE_SIZE, len(embeddings))
    sample_idx = rng.choice(len(embeddings), size=n_sample, replace=False)
    sample = embeddings[sample_idx]
    sample_coords = coords[sample_idx]
    print(f"Using random subsample of {n_sample} tiles for silhouette + PCA (O(n^2) cost).")

    scar_centres_4 = np.load(SCAR_CENTRES_4_PATH)
    scar_centres_5 = np.load(SCAR_CENTRES_5_PATH)
    assert scar_centres_4.shape[0] == 4
    assert scar_centres_5.shape[0] == 5

    macro_labels = nearest_cluster(sample, scar_centres_4)
    sub_labels = nearest_cluster(sample, scar_centres_5)

    gt_labels = lookup_gt_labels(sample_coords)

    # Fit PCA ONCE on `sample`, reuse the same 2D coords + axis limits for
    # every panel below (including the combined figure). Shared PCA
    # coordinates allow direct comparison across panels.
    coords_2d, explained, xlim, ylim = compute_shared_pca(sample)
    print(f"\nFor the figure caption: PC1 = {explained[0]*100:.1f}% of variance, "
          f"PC2 = {explained[1]*100:.1f}% "
          f"(total {sum(explained[:2])*100:.1f}%)")

    print("\n=== MACRO clusters (scar_centres_4: artefact/normal/scar/peri) ===")
    report_silhouette("macro", sample, macro_labels, MACRO_LABELS)
    pca_plot(coords_2d, explained, macro_labels, xlim, ylim,
             "Scar-candidate tiles, colored by macro cluster",
             OUTPUT_DIR / "cluster_separation_pca_macro", MACRO_LABELS,
             panel_letter="A")

    print("\n=== SUB clusters (scar_centres_5: sub-clusters within scar-candidates) ===")
    report_silhouette("sub", sample, sub_labels)
    pca_plot(coords_2d, explained, sub_labels, xlim, ylim,
             "Scar-candidate tiles, colored by sub cluster",
             OUTPUT_DIR / "cluster_separation_pca_sub", None,
             panel_letter="B")

    print("\n=== GROUND TRUTH labels (gt_scar_mask per slide) ===")
    report_silhouette("gt", sample, gt_labels, GT_LABEL_NAMES)
    pca_plot(coords_2d, explained, gt_labels, xlim, ylim,
             "Scar-candidate tiles, colored by ground truth",
             OUTPUT_DIR / "cluster_separation_pca_gt", GT_LABEL_NAMES,
             panel_letter="C", colors=GT_COLORS)

    print("\n=== Combined 3-panel figure (A: macro, B: sub, C: GT), stacked vertically ===")
    combined_pca_figure(coords_2d, explained, xlim, ylim,
                         macro_labels, sub_labels, gt_labels,
                         OUTPUT_DIR / "cluster_separation_pca_combined")

    print("\n=== Comparison ===")
    print("If MACRO silhouette >> SUB silhouette: the coarse artefact/normal/scar/peri split is more")
    print("  well-defined than the finer sub-clustering within scar -- sub-clusters may be")
    print("  splitting a genuinely uniform population, not real structure.")
    print("If SUB silhouette >> MACRO silhouette: sub-clusters ARE capturing real structure")
    print("  that the macro split misses.")
    print("If both are low (near 0): heavy overlap at both levels, matching the PCA plot --")
    print("  clustering (of either granularity) may not be a reliable lever for improving")
    print("  scar/non-scar separation on its own.")
    print("Compare GT silhouette to MACRO/SUB: if GT silhouette is also low, the embedding")
    print("  space itself does not cleanly separate GT-scar from GT-non-scar tiles at this")
    print("  granularity -- consistent with the classifier-confidence explanation for misses,")
    print("  rather than a clustering/filtering artefact.")


def collect_stratified_tissue_embeddings(n_per_class=500, rng=None):
    """Balanced sample: equal tiles per head-predicted class, per slide,
    so nearest-centroid assignment isn't dominated by class prevalence."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    embeddings_by_class = {0: [], 1: [], 2: [], 3: []}
    for slide_name in SLIDE_NAMES:
        head_path = MODEL_OUTPUT_DIR / f"{slide_name}_scar_head_arr.npy"
        norm_path = MODEL_OUTPUT_DIR / f"{slide_name}_scar_norm_arr.npy"
        if not (head_path.exists() and norm_path.exists()):
            continue
        head_arr = np.load(head_path)
        norm_arr = np.load(norm_path)
        has_tissue = np.any(head_arr != 0, axis=2)
        predicted_class = np.argmax(head_arr, axis=2)
        for c in range(4):
            idx = np.argwhere(has_tissue & (predicted_class == c))
            if len(idx) == 0:
                continue
            n_take = min(n_per_class, len(idx))
            chosen = rng.choice(len(idx), size=n_take, replace=False)
            idx = idx[chosen]
            embeddings_by_class[c].append(norm_arr[idx[:, 0], idx[:, 1], :])
    all_embeddings, all_head_class = [], []
    for c, chunks in embeddings_by_class.items():
        if chunks:
            e = np.concatenate(chunks, axis=0)
            all_embeddings.append(e)
            all_head_class.append(np.full(len(e), c))
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_head_class, axis=0)


def verify_macro_labels():
    """Cross-tabulates macro-cluster assignment (from scar_centres_4) against
    the classification head's own argmax class, to check whether centroid
    row order matches head-channel order (artefact/normal/scar/peri)."""
    embeddings, head_class = collect_stratified_tissue_embeddings()
    scar_centres_4 = np.load(SCAR_CENTRES_4_PATH)
    macro_labels = nearest_cluster(embeddings, scar_centres_4)

    import pandas as pd
    df = pd.DataFrame({
        "macro_cluster": macro_labels,
        "head_class": [MACRO_LABELS[c] for c in head_class],
    })
    ct = pd.crosstab(df["macro_cluster"], df["head_class"])
    print("\nMacro cluster vs. head-predicted class (row-normalised %):")
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1))
    print("\nRaw counts:")
    print(ct)
    return ct


if __name__ == "__main__":
    main()
    verify_macro_labels()