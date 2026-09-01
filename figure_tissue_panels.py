# python figure_tissue_panels.py
#
# Builds, for each slide, a 4x1 (stacked vertically) figure, each panel
# rotated to landscape orientation:
#   [1] Tissue, no annotation
#   [2] Tissue + Ground truth (red)
#   [3] Tissue + Algorithm output (green)
#   [4] Alignment of both (red = GT only, green = algo only, yellow = both)
#
# Cropped to the annotated region only (using each slide's *_valid_mask.npy,
# the same region 03_extract_ground_truth.py restricts its metrics to), with
# any leftover solid-black margin auto-trimmed so you see just the tissue.
#
# Run this after 02_classify_tiles.py and 03_extract_ground_truth.py have
# already produced their .npy outputs.

import re

import config

# OpenSlide's DLLs (Windows) must be locatable before `import openslide`.
config.configure_openslide()

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402
import matplotlib.pyplot as plt                             # noqa: E402
from openslide import OpenSlide                             # noqa: E402
from scipy import ndimage                                   # noqa: E402

from config import (                                        # noqa: E402
    SCAR_PREDICTION_DIR,
    GT_DIR as GROUND_TRUTH_DIR,
    SLIDE_DIR,                          # folder holding the raw slide files
    SLIDE_NAMES,
    TILE_SIZE,
)

OUTPUT_DIR = config.OUTPUT_DIR / "four_panel_figures"
MAX_THUMB_DIM = 1600        # max pixel dimension of the rendered tissue crop (speed/size tradeoff)
MARGIN_FRAC = 0.03          # white margin added around the trimmed tissue, as a fraction of its
                            # height/width. This padding is baked into the panel images themselves,
                            # so keep it small -- large values show up as dead space inside every
                            # panel that no figure-layout tweak can remove. Set to 0.0 for tightest.

# ---- figure text sizes (bumped up for paper-ready output) ------------------
TITLE_FONTSIZE = 22           # per-panel titles in the single-slide figures
SUPTITLE_FONTSIZE = 26        # slide name in the single-slide figures
COMBINED_TITLE_FONTSIZE = 20  # per-panel titles in the combined figure
COMBINED_LABEL_FONTSIZE = 26  # slide names in the combined figure

TITLE_PAD = 3                 # points between a title's baseline block and its axes
LINE_FACTOR = 1.65            # multiplier turning a font size into the vertical space it needs.
                              # Raise it if titles still crowd the image above them; lower it to
                              # squeeze the panels closer together.

# The landscape rotation below (np.rot90(p, k=1)) is applied identically to
# every slide, so it assumes the epidermis ends up on the correct side by
# default. That's not true for every slide -- add a slide's name here if,
# after the standard rotation, its epidermis lands on the bottom instead of
# the top; it'll get an extra top/bottom flip to correct it.
#
# Dataset-specific orientation correction: this set applies to the six slides
# used here and will need revising for a new cohort.
FLIP_VERTICALLY = {
    "MSW02_10_02_d70",
    "MSW02_11_10_d70",
    "MSW02_12_05_d70",
}

GT_COLOUR = (230, 0, 0)        # red
ALGO_COLOUR = (0, 170, 60)     # green
BOTH_COLOUR = (240, 210, 0)    # yellow
OVERLAY_ALPHA = 0.62


def text_height_inches(fontsize, pad_pt=TITLE_PAD):
    """Vertical space (in inches) to reserve for one line of text at the given
    point size, including the gap between it and the axes it labels.

    Matplotlib draws a title ABOVE the axes box, so any layout that doesn't
    explicitly reserve this much room lets the title spill onto whatever sits
    above it -- which is exactly how titles end up printed over the image of
    the panel above."""
    return (fontsize * LINE_FACTOR + pad_pt) / 72.0


# ---------------------------------------------------------------------------
def normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def find_slide_file(slide_name):
    """Locate the raw slide file matching slide_name inside SLIDE_DIR
    (searches recursively, matches common whole-slide-image extensions)."""
    exts = {".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".vms", ".vmu"}
    target = normalize(slide_name)
    for p in SLIDE_DIR.rglob("*"):
        if p.suffix.lower() in exts and target in normalize(p.stem):
            return p
    return None


def crop_bbox_from_valid_mask(valid_mask, pad_rows=0, pad_cols=0):
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        raise ValueError("valid_mask is empty -- no annotated region to crop to.")
    r_min = max(0, rows.min() - pad_rows)
    r_max = min(valid_mask.shape[0] - 1, rows.max() + pad_rows)
    c_min = max(0, cols.min() - pad_cols)
    c_max = min(valid_mask.shape[1] - 1, cols.max() + pad_cols)
    return r_min, r_max, c_min, c_max


def find_full_tissue_tile_bbox(slide_path, gt_r_min, gt_r_max, gt_c_min, gt_c_max,
                                grid_shape, tile_size=TILE_SIZE, thumb_max_dim=3000,
                                white_thresh=245, dilate_px=3, pad_tiles=3):
    """Use a low-res thumbnail of the WHOLE slide to find the true extent of
    the tissue piece that contains the GT annotation (via connected
    components), rather than guessing a fixed tile padding. Returns a tile
    grid bbox (r_min, r_max, c_min, c_max) covering that whole piece."""
    with OpenSlide(str(slide_path)) as slide:
        level0_w, level0_h = slide.dimensions
        thumb = slide.get_thumbnail((thumb_max_dim, int(thumb_max_dim * level0_h / level0_w)))
    thumb_arr = np.array(thumb.convert("RGB"))
    scale_x = thumb.size[0] / level0_w
    scale_y = thumb.size[1] / level0_h

    is_tissue = np.any(thumb_arr < white_thresh, axis=2)
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(is_tissue, structure=structure, iterations=dilate_px)
    labeled, num = ndimage.label(dilated)

    fallback = (gt_r_min, gt_r_max, gt_c_min, gt_c_max)
    if num == 0:
        return fallback

    # seed region in thumbnail coords = the GT tile bbox, converted to level-0
    # pixels (c * TILE_SIZE, no bounds offset -- see get_tissue_crop docstring)
    x0, y0 = gt_c_min * tile_size, gt_r_min * tile_size
    x1, y1 = (gt_c_max + 1) * tile_size, (gt_r_max + 1) * tile_size
    tx0, ty0 = max(0, int(x0 * scale_x)), max(0, int(y0 * scale_y))
    tx1 = min(thumb_arr.shape[1], int(x1 * scale_x))
    ty1 = min(thumb_arr.shape[0], int(y1 * scale_y))
    if tx1 <= tx0 or ty1 <= ty0:
        return fallback

    seed_labels = labeled[ty0:ty1, tx0:tx1]
    seed_tissue = is_tissue[ty0:ty1, tx0:tx1]
    candidates = seed_labels[seed_tissue & (seed_labels > 0)]
    if candidates.size == 0:
        return fallback
    vals, counts = np.unique(candidates, return_counts=True)
    best_label = vals[np.argmax(counts)]

    keep = (labeled == best_label) & is_tissue
    rows, cols = np.where(keep)
    if len(rows) == 0:
        return fallback

    lvl0_x0 = cols.min() / scale_x
    lvl0_x1 = (cols.max() + 1) / scale_x
    lvl0_y0 = rows.min() / scale_y
    lvl0_y1 = (rows.max() + 1) / scale_y

    r_min = int(lvl0_y0 // tile_size) - pad_tiles
    r_max = int(lvl0_y1 // tile_size) + pad_tiles
    c_min = int(lvl0_x0 // tile_size) - pad_tiles
    c_max = int(lvl0_x1 // tile_size) + pad_tiles

    # always include the GT bbox itself, and clip to the grid's actual shape
    r_min = max(0, min(r_min, gt_r_min))
    c_min = max(0, min(c_min, gt_c_min))
    r_max = min(grid_shape[0] - 1, max(r_max, gt_r_max))
    c_max = min(grid_shape[1] - 1, max(c_max, gt_c_max))
    return r_min, r_max, c_min, c_max


def get_tissue_crop(slide_path, r_min, r_max, c_min, c_max):
    """Read the tissue region corresponding to the given tile-grid bbox,
    at a resolution capped by MAX_THUMB_DIM, as an RGB PIL image.

    IMPORTANT: no bounds_x/bounds_y adjustment is applied here. In
    03_extract_ground_truth.py, c = (tile_x + bounds_x) // tile_size, and that
    same (tile_x + bounds_x) value is used directly as a pixel position on
    a thumbnail of
    the FULL, uncropped slide. That means (tile_x + bounds_x) already IS
    the absolute level-0 pixel coordinate -- so grid column/row index c/r
    already has the bounds offset baked in, and level0_x = c * TILE_SIZE
    with no further adjustment. Adding bounds_x again here would double
    count it and shift the crop.
    """
    x0 = c_min * TILE_SIZE
    y0 = r_min * TILE_SIZE
    x1 = (c_max + 1) * TILE_SIZE
    y1 = (r_max + 1) * TILE_SIZE
    region_w = x1 - x0
    region_h = y1 - y0

    with OpenSlide(str(slide_path)) as slide:
        downsample = max(1.0, max(region_w, region_h) / MAX_THUMB_DIM)
        level = slide.get_best_level_for_downsample(downsample)
        level_downsample = slide.level_downsamples[level]

        level_w = int(round(region_w / level_downsample))
        level_h = int(round(region_h / level_downsample))

        region = slide.read_region((int(x0), int(y0)), level, (level_w, level_h))
        region = region.convert("RGB")

    # final resize so the tissue image lines up pixel-for-pixel with an
    # integer number of pixels per tile (needed for clean mask upscaling)
    out_w = (c_max - c_min + 1) * 8   # 8 display px per tile
    out_h = (r_max - r_min + 1) * 8
    region = region.resize((out_w, out_h), Image.LANCZOS)
    return region


def trim_black_borders(rgb, masks, threshold=10, min_content_frac=0.02):
    """Trim solid-black rows/columns from the edges of a crop (e.g. from
    reading past the physical edge of the scanned tissue), keeping rgb
    and all mask arrays in `masks` (a dict of same-HxW boolean arrays)
    aligned to the trimmed result."""
    non_black = np.any(rgb > threshold, axis=2)
    row_frac = non_black.mean(axis=1)
    col_frac = non_black.mean(axis=0)
    rows_keep = np.where(row_frac > min_content_frac)[0]
    cols_keep = np.where(col_frac > min_content_frac)[0]
    if len(rows_keep) == 0 or len(cols_keep) == 0:
        return rgb, masks
    r0, r1 = rows_keep.min(), rows_keep.max() + 1
    c0, c1 = cols_keep.min(), cols_keep.max() + 1
    trimmed_rgb = rgb[r0:r1, c0:c1]
    trimmed_masks = {k: v[r0:r1, c0:c1] for k, v in masks.items()}
    return trimmed_rgb, trimmed_masks


def isolate_and_crop_main_tissue(rgb, masks, gt_mask, white_thresh=245, dilate_px=8,
                                  bbox_pad_px=15, open_px=4):
    """Identify the connected tissue component that actually contains the
    GT annotation, whiten out any other separate tissue fragments that
    happen to fall inside the (generously padded) crop, and crop tightly
    to that component's bounding box.

    Thin line-shaped artifacts (fold lines, scratches, debris trails) that
    are technically touching the main tissue mass are NOT dropped by a
    plain connected-components pass, since they share a label with the
    real tissue. To catch those too, connectivity is decided on an
    *opened* version of the mask (erode then dilate), which snaps thin
    (<= ~open_px wide) protrusions off the main blob before labeling.
    The main tissue's full-resolution boundary is then recovered by
    growing that solid "core" back out through is_tissue only, so real
    tissue detail isn't lost -- only the disconnected thin lines are.
    """
    is_tissue = np.any(rgb < white_thresh, axis=2)
    if not is_tissue.any():
        return rgb, masks

    structure = np.ones((3, 3), dtype=bool)

    # snap off thin line-like artifacts before deciding connectivity
    opened = ndimage.binary_opening(is_tissue, structure=structure, iterations=open_px)

    dilated = ndimage.binary_dilation(opened, structure=structure, iterations=dilate_px)
    labeled, num = ndimage.label(dilated)
    if num == 0:
        return rgb, masks

    if gt_mask.any():
        overlaps = ndimage.sum(gt_mask, labeled, index=np.arange(1, num + 1))
    else:
        overlaps = np.zeros(num)
    if overlaps.sum() > 0:
        best_label = int(np.argmax(overlaps)) + 1
    else:
        # fall back to the largest tissue component if GT doesn't overlap any
        sizes = ndimage.sum(opened, labeled, index=np.arange(1, num + 1))
        best_label = int(np.argmax(sizes)) + 1

    # regrow the solid core back out to the full tissue boundary, but only
    # through pixels that are actually tissue -- this recovers legitimate
    # detail on the main piece while leaving thin, separately-opened-off
    # lines and genuinely separate fragments un-recovered
    core = opened & (labeled == best_label)
    keep = ndimage.binary_dilation(core, structure=structure, iterations=dilate_px + open_px + 2,
                                    mask=is_tissue)

    stray = is_tissue & ~keep
    rgb_clean = rgb.copy()
    rgb_clean[stray] = 255
    masks_clean = {k: v & ~stray for k, v in masks.items()}

    if not keep.any():
        return rgb_clean, masks_clean
    rows, cols = np.where(keep)
    r0 = max(0, rows.min() - bbox_pad_px)
    r1 = min(rgb.shape[0], rows.max() + 1 + bbox_pad_px)
    c0 = max(0, cols.min() - bbox_pad_px)
    c1 = min(rgb.shape[1], cols.max() + 1 + bbox_pad_px)

    rgb_out = rgb_clean[r0:r1, c0:c1]
    masks_out = {k: v[r0:r1, c0:c1] for k, v in masks_clean.items()}
    return rgb_out, masks_out


def blacken_to_white(rgb, threshold=10):
    """Convert any remaining near-black pixels (no-data corners left over
    where the rectangular crop box doesn't cover a diagonal tissue strip)
    to white, so they blend into the background instead of standing out
    as black blocks."""
    out = rgb.copy()
    is_black = np.all(out < threshold, axis=2)
    out[is_black] = 255
    return out


def add_margin(rgb, masks, frac=MARGIN_FRAC):
    """Pad a white margin around the tissue crop so it sits with some
    breathing room inside its panel instead of touching the edges."""
    h, w, _ = rgb.shape
    pad_h, pad_w = int(h * frac), int(w * frac)
    if pad_h == 0 and pad_w == 0:
        return rgb, masks
    rgb_padded = np.pad(rgb, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                         mode="constant", constant_values=255)
    masks_padded = {
        k: np.pad(v, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=False)
        for k, v in masks.items()
    }
    return rgb_padded, masks_padded


def mask_to_pixel_size(mask_crop, size_wh):
    """Nearest-neighbour upscale a boolean tile-grid mask to pixel resolution."""
    img = Image.fromarray((mask_crop.astype(np.uint8) * 255))
    return np.array(img.resize(size_wh, Image.NEAREST)) > 0


def overlay(base_rgb, mask, colour, alpha=OVERLAY_ALPHA, on_gray=True):
    if on_gray:
        gray = np.array(Image.fromarray(base_rgb).convert("L"))
        base = np.stack([gray] * 3, axis=-1)
    else:
        base = base_rgb
    out = base.astype(np.float32).copy()
    colour_arr = np.array(colour, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * colour_arr
    return out.astype(np.uint8)


def build_alignment_panel(base_rgb, gt_mask, algo_mask, alpha=OVERLAY_ALPHA):
    gray = np.array(Image.fromarray(base_rgb).convert("L"))
    gray_rgb = np.stack([gray] * 3, axis=-1)
    out = gray_rgb.astype(np.float32).copy()

    gt_only = gt_mask & ~algo_mask
    algo_only = algo_mask & ~gt_mask
    both = gt_mask & algo_mask

    for mask, colour in [(gt_only, GT_COLOUR), (algo_only, ALGO_COLOUR), (both, BOTH_COLOUR)]:
        colour_arr = np.array(colour, dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * colour_arr

    return out.astype(np.uint8)


def overlap_score(a, b):
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
def build_slide_panels(slide_name):
    """Load a slide's data and build its 4 rotated/flipped panels (tissue,
    GT overlay, algorithm overlay, alignment) plus the IoU score. Returns
    None if the slide's files can't be found (already prints why), so
    callers can just skip it. Shared by make_figure and
    make_combined_figure so the image-processing pipeline lives in one
    place.
    """
    try:
        algo_grid = np.load(SCAR_PREDICTION_DIR / f"{slide_name}_scar_prediction.npy") > 0
        gt_grid = np.load(GROUND_TRUTH_DIR / f"{slide_name}_gt_scar_mask.npy")
        valid_mask = np.load(config.valid_mask_path(slide_name))
    except FileNotFoundError as e:
        print(f"  SKIPPED {slide_name} -- missing file: {e}")
        return None

    gt_r_min, gt_r_max, gt_c_min, gt_c_max = crop_bbox_from_valid_mask(valid_mask)

    slide_path = find_slide_file(slide_name)
    if slide_path is None:
        print(f"  SKIPPED -- could not locate slide file for {slide_name} in {SLIDE_DIR}")
        return None

    # expand the GT-only bbox to cover the FULL physical tissue piece it sits
    # within, detected from a whole-slide thumbnail (not a guessed padding)
    r_min, r_max, c_min, c_max = find_full_tissue_tile_bbox(
        slide_path, gt_r_min, gt_r_max, gt_c_min, gt_c_max, grid_shape=gt_grid.shape
    )

    algo_crop = algo_grid[r_min:r_max + 1, c_min:c_max + 1]
    gt_crop = gt_grid[r_min:r_max + 1, c_min:c_max + 1]

    tissue_img = get_tissue_crop(slide_path, r_min, r_max, c_min, c_max)
    tissue_rgb = np.array(tissue_img)
    size_wh = tissue_img.size  # (w, h)

    gt_px = mask_to_pixel_size(gt_crop, size_wh)
    algo_px = mask_to_pixel_size(algo_crop, size_wh)

    # drop any solid-black margin left over from reading past the tissue edge
    tissue_rgb, trimmed = trim_black_borders(tissue_rgb, {"gt": gt_px, "algo": algo_px})
    gt_px, algo_px = trimmed["gt"], trimmed["algo"]

    # any remaining no-data corners (rectangular crop vs. diagonal tissue) -> white
    tissue_rgb = blacken_to_white(tissue_rgb)

    # isolate the tissue component that actually contains the GT annotation,
    # discard any other separate tissue fragments swept in by the generous
    # padding, and crop tightly to just that component
    tissue_rgb, isolated = isolate_and_crop_main_tissue(tissue_rgb, {"gt": gt_px, "algo": algo_px}, gt_px)
    gt_px, algo_px = isolated["gt"], isolated["algo"]

    # add a bit of white breathing room so the tissue reads smaller within each panel
    tissue_rgb, padded = add_margin(tissue_rgb, {"gt": gt_px, "algo": algo_px})
    gt_px, algo_px = padded["gt"], padded["algo"]

    panel1 = tissue_rgb
    panel2 = overlay(tissue_rgb, gt_px, GT_COLOUR)
    panel3 = overlay(tissue_rgb, algo_px, ALGO_COLOUR)
    panel4 = build_alignment_panel(tissue_rgb, gt_px, algo_px)

    # rotate to landscape (tissue runs horizontally within each panel)
    panel1, panel2, panel3, panel4 = (np.rot90(p, k=1) for p in (panel1, panel2, panel3, panel4))

    # correct epidermis-down slides (see FLIP_VERTICALLY above)
    if slide_name in FLIP_VERTICALLY:
        panel1, panel2, panel3, panel4 = (np.flipud(p) for p in (panel1, panel2, panel3, panel4))

    score = overlap_score(gt_crop, algo_crop)
    return panel1, panel2, panel3, panel4, score


def make_figure(slide_name):
    print(f"\n{'='*60}\n{slide_name}")

    panels = build_slide_panels(slide_name)
    if panels is None:
        return
    panel1, panel2, panel3, panel4, score = panels

    # ---- layout -----------------------------------------------------------
    # Everything below is computed in INCHES, then converted to figure
    # fractions at the end. Two problems are solved at once:
    #
    #  (a) Trapped white space. imshow keeps the image's aspect ratio, so an
    #      axes box taller than the (wide, landscape) image letterboxes it and
    #      the leftover height becomes dead space INSIDE the axes that no
    #      hspace or bbox_inches setting can reach. So each image row is given
    #      exactly the height its aspect ratio needs: img_h = img_w / aspect.
    #
    #  (b) Titles landing on the image above. A title is drawn above its axes
    #      box, and hspace is a fraction of the average axes height -- so a
    #      small hspace silently gives the title less room than it needs. Here
    #      hspace is 0 and each title instead gets its own dedicated spacer
    #      row, sized in inches from the font size. A title physically cannot
    #      overlap the panel above it.
    h_px, w_px, _ = panel1.shape
    aspect = w_px / h_px

    img_w = 10.0
    img_h = max(1.2, img_w / aspect)
    title_h = text_height_inches(TITLE_FONTSIZE)
    suptitle_h = text_height_inches(SUPTITLE_FONTSIZE, pad_pt=6)
    side_h, bottom_h = 0.12, 0.12

    fig_w = img_w + 2 * side_h
    fig_h = suptitle_h + 4 * (title_h + img_h) + bottom_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # rows alternate: [space for title][image][space for title][image]...
    ratios = []
    for _ in range(4):
        ratios += [title_h, img_h]
    gs = fig.add_gridspec(
        8, 1, height_ratios=ratios, hspace=0.0,
        left=side_h / fig_w, right=1 - side_h / fig_w,
        top=1 - suptitle_h / fig_h, bottom=bottom_h / fig_h,
    )

    titles = [
        "Tissue",
        "Ground truth",
        "Algorithm",
        f"Alignment (IoU = {score:.2f})",
    ]
    for i, (img, title) in enumerate(zip([panel1, panel2, panel3, panel4], titles)):
        ax = fig.add_subplot(gs[2 * i + 1, 0])
        ax.imshow(img)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
        ax.axis("off")

    fig.suptitle(slide_name, fontsize=SUPTITLE_FONTSIZE, va="top", y=0.998)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{slide_name}_4panel.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}  (IoU in annotated region: {score:.3f})")


def make_combined_figure():
    """One paper-ready figure containing all 6 slides: two slides per row,
    each slide shown as its own 4-panel vertical strip (tissue, GT,
    algorithm, alignment) -- i.e. the same 4-panel content as
    make_figure(), just laid out 2-per-row instead of one file per slide.
    """
    print(f"\n{'='*60}\nCombined figure (all slides)")

    results = []
    for slide_name in SLIDE_NAMES:
        panels = build_slide_panels(slide_name)
        if panels is None:
            continue
        results.append((slide_name, panels))

    if not results:
        print("  SKIPPED combined figure -- no slides available")
        return

    ncols = 2
    nrows = int(np.ceil(len(results) / ncols))

    # Same inch-based scheme as make_figure. The panel height comes from the
    # median aspect across slides; a slide that's relatively taller than the
    # median simply renders a little narrower inside its box, and one that's
    # wider letterboxes slightly -- neither can collide with a title, because
    # every title owns its own spacer row.
    aspect = float(np.median([res[1][0].shape[1] / res[1][0].shape[0] for res in results]))

    img_w = 5.6
    img_h = max(0.8, img_w / aspect)
    title_h = text_height_inches(COMBINED_TITLE_FONTSIZE)
    label_h = text_height_inches(COMBINED_LABEL_FONTSIZE, pad_pt=4)
    row_gap = 0.45          # vertical space between slide groups
    col_gap = 0.35          # horizontal space between the two columns
    side_h, top_h, bottom_h = 0.12, 0.12, 0.12

    cell_w = img_w
    cell_h = label_h + 4 * (title_h + img_h)

    fig_w = ncols * cell_w + (ncols - 1) * col_gap + 2 * side_h
    fig_h = nrows * cell_h + (nrows - 1) * row_gap + top_h + bottom_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        nrows, ncols, hspace=row_gap / cell_h, wspace=col_gap / cell_w,
        left=side_h / fig_w, right=1 - side_h / fig_w,
        top=1 - top_h / fig_h, bottom=bottom_h / fig_h,
    )

    panel_titles = ["Tissue", "Ground truth", "Algorithm", "Alignment"]

    for idx, (slide_name, (p1, p2, p3, p4, score)) in enumerate(results):
        r, c = divmod(idx, ncols)

        # first row holds the slide-name label AND the first panel's title;
        # after that, rows alternate title-space / image
        inner_ratios = [label_h + title_h]
        for _ in range(4):
            inner_ratios += [img_h, title_h]
        inner_ratios = inner_ratios[:-1]          # no trailing title row
        inner = outer[r, c].subgridspec(len(inner_ratios), 1,
                                        height_ratios=inner_ratios, hspace=0.0)

        titles = panel_titles[:3] + [f"Alignment (IoU = {score:.2f})"]
        for panel_idx, (img, title) in enumerate(zip((p1, p2, p3, p4), titles)):
            ax = fig.add_subplot(inner[2 * panel_idx + 1, 0])
            ax.imshow(img)
            ax.set_title(title, fontsize=COMBINED_TITLE_FONTSIZE, pad=TITLE_PAD)
            ax.axis("off")

        # slide name sits in the label_h slot at the very top of this cell
        cell_pos = outer[r, c].get_position(fig)
        fig.text((cell_pos.x0 + cell_pos.x1) / 2, cell_pos.y1, slide_name,
                  ha="center", va="top", fontsize=COMBINED_LABEL_FONTSIZE,
                  fontweight="bold")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "all_slides_4panel.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    for slide_name in SLIDE_NAMES:
        try:
            make_figure(slide_name)
        except FileNotFoundError as e:
            print(f"  SKIPPED {slide_name} -- missing file: {e}")

    try:
        make_combined_figure()
    except FileNotFoundError as e:
        print(f"  SKIPPED combined figure -- missing file: {e}")


if __name__ == "__main__":
    main()