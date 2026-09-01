"""How narrowly did the missed tiles lose? (Methods 2.5)

    python analysis_missed_scar_margin.py

Drops the embedding entirely and works from the classification head's raw
four-channel scores. For every false negative where a class other than
scar won, this computes

    margin = winning channel's score - scar channel's score

along with which channel won. A small margin means scar was a close
runner-up and a softer threshold could plausibly recover the tile. A large
margin means scar was decisively beaten, which no threshold change will
fix.

This matters because the embedding-based reading is not dependable on its
own: analysis_missed_scar_centroid_check.py shows the macro-centroid
labelling holds for only 59.7% of confirmed true positives.

The reported figures are from the baseline arm, so run with
SCAR_EVALUATION=baseline.
"""

import csv

import numpy as np

import config
from config import (
    MODEL_OUTPUT_DIR,
    SCAR_PREDICTION_DIR,
    SCAR_CLASS_INDEX,
    SLIDE_NAMES,
)
from metrics import load_outcome_masks

OUTPUT_DIR = config.OUTPUT_DIR / "missed_scar_analysis" / config.EVALUATION
OUT_CSV = OUTPUT_DIR / "missed_scar_margin.csv"

CHANNEL_NAMES = {0: "arte", 1: "norm", 2: "scar", 3: "peri"}


def main():
    config.ensure_dirs(OUTPUT_DIR)
    print(f"SCAR_EVALUATION = '{config.EVALUATION}'\n")
    all_margins = []
    all_winning_channels = []
    per_slide_rows = []

    for slide in SLIDE_NAMES:
        head_arr = np.load(MODEL_OUTPUT_DIR / f"{slide}_scar_head_arr.npy")

        gt, algo, fn_mask = load_outcome_masks(
            slide, SCAR_PREDICTION_DIR / f"{slide}_scar_prediction.npy")

        n_fn = int(fn_mask.sum())

        fn_scores = head_arr[fn_mask]              # (n_fn, 4)
        winning_channel = np.argmax(fn_scores, axis=1)
        max_score = fn_scores.max(axis=1)
        scar_score = fn_scores[:, SCAR_CLASS_INDEX]
        margin = max_score - scar_score             # >=0 always, since scar lost (or tied)

        # only tiles where scar genuinely lost (winning_channel != 2) --
        # the same "not a candidate" group analysis_missed_scar_stage.py isolates
        not_candidate = winning_channel != SCAR_CLASS_INDEX
        margin_nc = margin[not_candidate]
        winning_nc = winning_channel[not_candidate]

        all_margins.append(margin_nc)
        all_winning_channels.append(winning_nc)

        if len(margin_nc) > 0:
            mean_m = float(margin_nc.mean())
            median_m = float(np.median(margin_nc))
            p10_m = float(np.percentile(margin_nc, 10))
            p25_m = float(np.percentile(margin_nc, 25))
        else:
            mean_m = median_m = p10_m = p25_m = float("nan")

        winner_counts = {CHANNEL_NAMES[c]: int((winning_nc == c).sum()) for c in range(4) if c != SCAR_CLASS_INDEX}

        per_slide_rows.append({
            "slide": slide,
            "n_fn": n_fn,
            "n_fn_not_candidate": int(not_candidate.sum()),
            "margin_mean": round(mean_m, 4),
            "margin_median": round(median_m, 4),
            "margin_p10": round(p10_m, 4),
            "margin_p25": round(p25_m, 4),
            **{f"winner_{k}": v for k, v in winner_counts.items()},
        })

        print(f"{slide}: {len(margin_nc)} FN tiles where scar lost outright")
        print(f"    margin (winning score - scar score): mean={mean_m:.4f} "
              f"median={median_m:.4f} p10={p10_m:.4f} p25={p25_m:.4f}")
        print(f"    winning channel breakdown: {winner_counts}")

    pooled_margins = np.concatenate(all_margins) if all_margins else np.array([])
    pooled_winners = np.concatenate(all_winning_channels) if all_winning_channels else np.array([])

    print("\n=== POOLED ===")
    if len(pooled_margins) > 0:
        print(f"n = {len(pooled_margins)} FN tiles where scar lost outright")
        print(f"margin: mean={pooled_margins.mean():.4f}  median={np.median(pooled_margins):.4f}")
        for p in (5, 10, 25, 50, 75):
            print(f"  p{p} = {np.percentile(pooled_margins, p):.4f}")
        winner_counts_pooled = {CHANNEL_NAMES[c]: int((pooled_winners == c).sum())
                                 for c in range(4) if c != SCAR_CLASS_INDEX}
        print(f"winning channel breakdown (pooled): {winner_counts_pooled}")

        # rough close-call threshold: bottom 10% of margins, pooled
        close_cutoff = np.percentile(pooled_margins, 10)
        n_close = int((pooled_margins <= close_cutoff).sum())
        print(f"\n~{n_close} tiles ({100*n_close/len(pooled_margins):.1f}%) have margin <= "
              f"{close_cutoff:.4f} (bottom 10%, pooled) -- these are the closest calls.")

    fieldnames = list(per_slide_rows[0].keys())
    # collect full set of fieldnames across slides in case winner channel keys differ
    all_keys = set()
    for row in per_slide_rows:
        all_keys.update(row.keys())
    fieldnames = list(all_keys)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_slide_rows:
            writer.writerow(row)
    print(f"\nSaved per-slide summary to {OUT_CSV} (only new file written, nothing else touched)")


if __name__ == "__main__":
    main()
