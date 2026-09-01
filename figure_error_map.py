"""Per-tile spatial error map (Methods 2.5).

    python figure_error_map.py

Labels every tile TP, FP, FN, TN or background and plots it in its original
spatial position, to check whether the errors are spatially concentrated --
at tissue margins, say -- rather than scattered at random.

This is the abstract grid view. The same information composited onto the
real tissue image is one of the outputs of 03_extract_ground_truth.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
from config import SCAR_PREDICTION_DIR, SLIDE_NAMES as SLIDE_IDS
from metrics import load_scored_region

OUTPUT_DIR = config.OUTPUT_DIR / "error_maps" / config.EVALUATION

# 0=background/out-of-region, 1=TN, 2=TP, 3=FP, 4=FN
COLORS = ["black", "lightgray", "green", "red", "blue"]
CMAP = ListedColormap(COLORS)
LABELS = ["background/invalid", "TN", "TP", "FP", "FN"]


def build_error_grid(slide_id):
    algo, gt, region_mask, _ = load_scored_region(
        slide_id, SCAR_PREDICTION_DIR / f"{slide_id}_scar_prediction.npy")

    algo_scar = algo & region_mask
    gt_scar = gt & region_mask

    grid = np.zeros(region_mask.shape, dtype=np.int32)
    grid[region_mask & ~algo_scar & ~gt_scar] = 1  # TN
    grid[algo_scar & gt_scar] = 2                   # TP
    grid[algo_scar & ~gt_scar] = 3                  # FP
    grid[~algo_scar & gt_scar] = 4                  # FN
    return grid


def main():
    config.ensure_dirs(OUTPUT_DIR)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'\n")
    for slide_id in SLIDE_IDS:
        try:
            grid = build_error_grid(slide_id)
        except FileNotFoundError as e:
            print(f"{slide_id}: SKIPPED - {e}")
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(f"{slide_id}\ngreen=TP  red=FP  blue=FN  light gray=TN  black=background/invalid")
        ax.axis("off")

        out_path = OUTPUT_DIR / f"{slide_id}_error_map.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
