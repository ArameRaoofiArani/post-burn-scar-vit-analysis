# python figure_subcluster_maps.py
#
# Step 3 (figure). Sub-cluster maps from scar_centres_5: same colours and
# layout as 02_classify_tiles.py's preview, with each band named by the
# sub-cluster it represents. Produces both
#
#   - one image per slide, and
#   - one combined plate: 2 rows x 3 columns, panels labelled a) to f),
#     with the sub-cluster colourbar down the right-hand side.
#
# Tile values are nearest-centroid assignments stored as cluster_index + 1:
#
#     value 0  black       not scar / background
#     value 1  khaki       sub-cluster 0
#     value 2  limegreen   sub-cluster 1
#     value 3  teal        sub-cluster 2
#     value 4  royalblue   sub-cluster 3
#     value 5  orangered   sub-cluster 4
#
# These are categorical cluster identities, not an ordered severity scale --
# nothing here grades severity, which is why the colourbar bands are named
# rather than left as bare integers (the integers are cluster_index + 1,
# which no reader can guess).
#
# SHOW_ALL_SUBCLUSTERS controls what is drawn:
#   True  -- all five, including 0 and 1. Those two are the ones
#            02_classify_tiles.py discards as non-scar, so this shows what
#            the scar-candidate gate caught BEFORE the sub-cluster filter.
#   False -- original behaviour: 0 and 1 render as background, so only the
#            retained scar (2, 3, 4) is visible.
#
# Only the tissue section carrying the ground-truth annotation is shown;
# the other sections on each slide were never annotated.
#
# Writes into its own timestamped folder; nothing existing is overwritten.

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from scipy import ndimage
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
import config                           # noqa: E402
from config import (                    # noqa: E402
    MODEL_OUTPUT_DIR,
    GT_DIR,
    OUTPUT_DIR as FIGURE_DIR,
    SCAR_CENTRES_5_PATH as SCAR_CENTRES_PATH,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
    VALID_CLUSTER_INDICES,             # retained as scar by 02_classify_tiles.py
    CLUSTER_COLOURS,
)

SHOW_ALL_SUBCLUSTERS = True             # see note in the header

# ---- output options -------------------------------------------------------
MAKE_SINGLE_PREVIEWS = True
MAKE_COMBINED_PANEL = True
CROP_TO_ANNOTATED_PIECE = True          # applies to both outputs
CROP_PAD = 5
TISSUE_MIN_COMPONENT = 150              # lower to ~40 if a torn section
                                        # loses one of its fragments

NCOLS, NROWS = 3, 2
PANEL_HEIGHT_IN = 5.4
COLOURBAR_COLUMN_IN = 2.6               # reserved width for the colourbar
COLOURBAR_WIDTH_IN = 0.32
COLOURBAR_FRAC = 0.62                   # height, as a fraction of the plate

PANEL_LETTERS = "abcdef"

OUTPUT_DIR = FIGURE_DIR / f"subcluster_previews_{datetime.now():%Y%m%d_%H%M%S}"

# identical to 02_classify_tiles.py
COLOURS = CLUSTER_COLOURS
CMAP = ListedColormap(COLOURS)

TICK_LABELS = ["not scar", "sub-cluster 0", "sub-cluster 1",
               "sub-cluster 2", "sub-cluster 3", "sub-cluster 4"]

TITLE_FS = 14
CBAR_FS = 13
SAVE_DPI = 300
SAVE_PDF = True

STRUCT8 = np.ones((3, 3), dtype=bool)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
def build_subcluster_grid(slide_name, scar_centres):
    """Assigns every scar candidate to its nearest of the five centroids.

    Vectorised with cdist rather than looping per tile as
    02_classify_tiles.py does -- identical result, far faster.
    """
    head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_head_arr.npy")
    norm_arr = np.load(MODEL_OUTPUT_DIR / f"{slide_name}_scar_norm_arr.npy")

    has_tissue = np.any(head_arr != 0, axis=2)
    predicted_class = np.argmax(head_arr, axis=2)
    is_scar = (predicted_class == SCAR_CLASS_INDEX) & has_tissue
    tissue_mask = head_arr.sum(axis=-1) != 0

    grid = np.zeros(head_arr.shape[:2], dtype=np.int32)
    r, c = np.where(is_scar)
    if len(r) == 0:
        return grid, {}, tissue_mask

    nearest = np.argmin(cdist(norm_arr[r, c, :], scar_centres), axis=1)

    if SHOW_ALL_SUBCLUSTERS:
        keep = np.ones(len(nearest), dtype=bool)
    else:
        keep = np.isin(nearest, list(VALID_CLUSTER_INDICES))

    grid[r[keep], c[keep]] = nearest[keep] + 1

    counts = {int(cid): int((nearest == cid).sum())
              for cid in range(scar_centres.shape[0])}
    return grid, counts, tissue_mask


def clean_tissue_mask(tissue, min_component=TISSUE_MIN_COMPONENT):
    """head_arr.sum(-1) != 0 flags every tile the model wrote output for,
    including single-tile-wide streaks at scan boundaries and small label
    blocks, which otherwise appear as thin stair-stepped outlines around
    the section. Opening snaps them off; surviving cores are regrown
    through the original mask so genuine tissue detail is kept."""
    opened = ndimage.binary_opening(tissue, structure=STRUCT8, iterations=1)
    lab, n = ndimage.label(opened, structure=STRUCT8)
    if n == 0:
        return tissue
    sizes = ndimage.sum(opened, lab, index=np.arange(1, n + 1))
    keep_labels = np.where(sizes >= min_component)[0] + 1
    if len(keep_labels) == 0:
        return tissue
    core = np.isin(lab, keep_labels)
    return ndimage.binary_dilation(core, structure=STRUCT8, iterations=2,
                                   mask=tissue)


def annotated_piece(tissue, gt_mask, valid_mask):
    """The tissue component carrying the ground truth, chosen by GT overlap
    rather than size -- the annotated piece is not always the largest
    fragment on the slide."""
    lab, n = ndimage.label(tissue, structure=STRUCT8)
    if n == 0:
        return tissue
    overlap = ndimage.sum(gt_mask, lab, index=np.arange(1, n + 1))
    if overlap.max() == 0:
        overlap = ndimage.sum(valid_mask, lab, index=np.arange(1, n + 1))
    if overlap.max() == 0:
        return tissue
    return lab == (int(np.argmax(overlap)) + 1)


def crop_to_annotated(grid, tissue_mask, slide_name):
    """Blanks every tissue piece except the annotated one and crops to it."""
    try:
        gt = np.load(GT_DIR / f"{slide_name}_gt_scar_mask.npy").astype(bool)
        valid = np.load(config.valid_mask_path(slide_name)).astype(bool)
        piece = annotated_piece(clean_tissue_mask(tissue_mask), gt, valid)
    except FileNotFoundError:
        piece = clean_tissue_mask(tissue_mask)

    out = np.where(piece, grid, 0)
    r, c = np.where(piece)
    if len(r) == 0:
        return out
    r0 = max(0, r.min() - CROP_PAD)
    r1 = min(grid.shape[0], r.max() + 1 + CROP_PAD)
    c0 = max(0, c.min() - CROP_PAD)
    c1 = min(grid.shape[1], c.max() + 1 + CROP_PAD)
    return out[r0:r1, c0:c1]


def pad_to(arr, target_h, target_w, fill=0):
    """Centres an array on a common canvas. Padding rather than rescaling
    keeps one tile the same physical size in every panel, so the sections
    stay comparable in scale."""
    h, w = arr.shape
    top = (target_h - h) // 2
    left = (target_w - w) // 2
    return np.pad(arr, ((top, target_h - h - top), (left, target_w - w - left)),
                  mode="constant", constant_values=fill)


def add_colourbar(fig, x0_in, fig_w, fig_h):
    """Discrete colourbar with one named band per sub-cluster.

    BoundaryNorm gives equal-height bands with the tick centred in each,
    rather than the continuous ramp a default colourbar would draw across
    what are categorical cluster identities.
    """
    bounds = np.arange(-0.5, 6.5, 1.0)
    norm = BoundaryNorm(bounds, CMAP.N)

    cb_h = COLOURBAR_FRAC
    cax = fig.add_axes([x0_in / fig_w, (1 - cb_h) / 2,
                        COLOURBAR_WIDTH_IN / fig_w, cb_h])
    cb = ColorbarBase(cax, cmap=CMAP, norm=norm, boundaries=bounds,
                      ticks=[0, 1, 2, 3, 4, 5], spacing="proportional")
    labels = list(TICK_LABELS)
    for cid in range(5):
        if cid not in VALID_CLUSTER_INDICES:
            labels[cid + 1] += "  (excluded)"
    cb.ax.set_yticklabels(labels, fontsize=CBAR_FS)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    return cb


# ---------------------------------------------------------------------------
def save_preview(grid, slide_name, out_dir):
    """Single-slide image, matching 02_classify_tiles.py's preview."""
    excluded = sorted(set(range(5)) - VALID_CLUSTER_INDICES)

    fig = plt.figure(figsize=(6, 12))
    ax = fig.add_subplot(111)
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    if SHOW_ALL_SUBCLUSTERS:
        subtitle = (f"all five sub-clusters shown; "
                    f"{excluded[0]} and {excluded[1]} are excluded from scar")
    else:
        subtitle = f"sub-clusters {excluded[0]} and {excluded[1]} excluded"
    ax.set_title(f"{slide_name}\n({subtitle})")

    bounds = np.arange(-0.5, 6.5, 1.0)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(bounds, CMAP.N),
                                            cmap=CMAP),
                      ax=ax, boundaries=bounds, ticks=[0, 1, 2, 3, 4, 5],
                      spacing="proportional")
    cb.ax.set_yticklabels(TICK_LABELS)

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{slide_name}_subcluster_preview"
    fig.savefig(stem.with_suffix(".png"), dpi=SAVE_DPI)
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem


def make_combined_panel(panels, out_dir):
    """2 x 3 plate, panels labelled a) to f), colourbar on the right.

    Panels carry letters rather than slide names: the plate stays legible
    at column width, and the slide IDs belong in the caption where they
    can be typeset and edited at proof.
    """
    max_h = max(g.shape[0] for _, g in panels)
    max_w = max(g.shape[1] for _, g in panels)
    grids = [(sid, pad_to(g, max_h, max_w)) for sid, g in panels]

    aspect = max_w / max_h
    cell_w = PANEL_HEIGHT_IN * aspect

    # Axes are placed in inches: a gridspec cell taller than its image
    # letterboxes the panel, and the leftover height becomes dead space
    # inside the axes that no spacing setting can remove.
    left_in, right_in = 0.20, 0.20
    top_in, bottom_in = 0.35, 0.35
    hgap_in, vgap_in = 0.30, 0.55
    title_in = 0.36

    grid_w = NCOLS * cell_w + (NCOLS - 1) * hgap_in
    fig_w = left_in + grid_w + COLOURBAR_COLUMN_IN + right_in
    fig_h = (top_in + NROWS * (PANEL_HEIGHT_IN + title_in)
             + (NROWS - 1) * vgap_in + bottom_in)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    for i, (slide_name, grid) in enumerate(grids):
        row, col = divmod(i, NCOLS)
        x0 = left_in + col * (cell_w + hgap_in)
        y_top = top_in + row * (PANEL_HEIGHT_IN + title_in + vgap_in) + title_in
        y0 = fig_h - y_top - PANEL_HEIGHT_IN

        ax = fig.add_axes([x0 / fig_w, y0 / fig_h,
                           cell_w / fig_w, PANEL_HEIGHT_IN / fig_h])
        ax.imshow(grid, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{PANEL_LETTERS[i]})", fontsize=TITLE_FS,
                     loc="left", pad=5)

    add_colourbar(fig, left_in + grid_w + 0.35, fig_w, fig_h)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "subclusters_all_slides"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                    bbox_inches="tight")
    plt.close(fig)
    return stem


# ---------------------------------------------------------------------------
def main():
    scar_centres = np.load(SCAR_CENTRES_PATH)
    assert scar_centres.shape[0] == 5, (
        f"expected 5 centroids in {SCAR_CENTRES_PATH}, got {scar_centres.shape[0]}"
    )
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"SHOW_ALL_SUBCLUSTERS = {SHOW_ALL_SUBCLUSTERS}\n")

    panels = []
    for slide_name in SLIDE_NAMES:
        print(f"Processing {slide_name} ...")
        try:
            grid, counts, tissue_mask = build_subcluster_grid(
                slide_name, scar_centres)
        except FileNotFoundError as e:
            print(f"  SKIPPED - file not found: {e}")
            continue

        shown = (crop_to_annotated(grid, tissue_mask, slide_name)
                 if CROP_TO_ANNOTATED_PIECE else grid)

        if MAKE_SINGLE_PREVIEWS:
            stem = save_preview(shown, slide_name, OUTPUT_DIR)
            print(f"  saved: {stem.name}.png")
        if MAKE_COMBINED_PANEL:
            panels.append((slide_name, shown))

        total = sum(counts.values())
        if total:
            for cid in sorted(counts):
                tag = "" if cid in VALID_CLUSTER_INDICES else "   (excluded)"
                print(f"    sub-cluster {cid}: {counts[cid]:>7,} tiles "
                      f"({100*counts[cid]/total:5.1f}%){tag}")
            kept = sum(n for cid, n in counts.items()
                       if cid in VALID_CLUSTER_INDICES)
            print(f"    retained as scar: {kept:,} of {total:,} candidates")
        print()

    if MAKE_COMBINED_PANEL and panels:
        stem = make_combined_panel(panels, OUTPUT_DIR)
        print(f"Combined panel saved: {stem.name}.png / .pdf")
        print("\nPanel order for the caption:")
        for i, (slide_name, _) in enumerate(panels):
            print(f"  {PANEL_LETTERS[i]}) {slide_name}")


if __name__ == "__main__":
    main()