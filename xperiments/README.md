[README.md](https://github.com/user-attachments/files/31701432/README.md)
# Experiments

Alternative classification rules that were tried and measured. None of them
feeds the main pipeline: each writes its own `*_scar_prediction_taskN.npy` and
leaves `<slide>_scar_prediction.npy` alone.

Run them from the repository root, e.g. `python experiments/task4_overlap_check.py`.

| Script | Question it asks |
| --- | --- |
| `task2_binary_metrics.py` | Same scoring as `04_evaluate_agreement.py`, reported with the 0 = scar / 1 = rest convention, plus per-slide 0/1 label grids on disk. |
| `task3_peripheral_clusters.py` | Does adding one cluster of the model's "peri" class recover scar the scar channel missed? Needs `peri_centres_4.npy`. |
| `task4_outlier_subclusters.py` | Does pruning IsolationForest outliers *within* each retained sub-cluster remove false positives? |
| `task4_overlap_check.py` | Scores the Task 4 grids against the baseline. |
| `task5_outlier_macroclusters.py` | Does reclassifying with the four macro centroids, then pruning outliers, beat the sub-cluster rule? Also computes a per-tile boundary margin. |
| `task5_overlap_check.py` | Scores the Task 5 grids against the baseline. |

## Read this before trusting the before/after numbers

Both overlap checks compare against `<slide>_scar_prediction.npy` as the "before"
column. That file is now written by the percentile-threshold rule in
`02_classify_tiles.py`, whereas Tasks 4 and 5 were developed when it held the
older `argmax == scar` output.

So the baseline column no longer isolates what these experiments changed — it
folds in a change of classification rule as well.

This is now fixable: run `SCAR_EVALUATION=baseline python 02_classify_tiles.py`
first, which writes an argmax-rule grid to `scar_predictions/baseline/`, then run
the experiment and its check under the same setting. The comparison then
isolates the outlier removal alone.

Relatedly, both Task 4 and Task 5 originally wrote an argmax-rule baseline to
`<slide>_scar_prediction.npy` if none existed. That would silently overwrite the
pipeline's output with a different classifier, so it now prints a message and
writes nothing instead.

## Where the macro-gated rule lives

Results 3.2 compares two gated rules: `argmax == scar` gated by the four
macro centroids, and the same gated by the five sub-cluster centroids. The
second is the pipeline baseline (`SCAR_EVALUATION=baseline`). The first has
no script of its own — it is what `task5_outlier_macroclusters.py` writes as
`<slide>_scar_prediction_task5_raw.npy`, before its outlier pruning, and what
`task5_overlap_check.py` reports in its `task5_raw` column. Verified to
reproduce the rule exactly.

To regenerate those figures:

```bash
SCAR_EVALUATION=baseline python experiments/task5_outlier_macroclusters.py
SCAR_EVALUATION=baseline python experiments/task5_overlap_check.py
```

The baseline arm, because 3.2 was scored over the bounding-box region.

Note also that the third rule named in Methods 2.2 — retaining all tiles
whose argmax is scar, ungated — has no metrics reported against it. It
defines the candidate population (94,416 tiles pooled) rather than being a
scored rule, and that count falls out of any script that builds the
candidate mask.

## The complete version

`../analysis_outlier_removal.py` supersedes the Task 4 and Task 5
outlier arms here. Those run at `contamination="auto"` only and use
IsolationForest alone; the full evaluation covers both detectors at three
contamination levels and both cluster levels, adds a random-removal control,
and measures the spatial structure of what was removed. The Task 4 and 5
scripts are kept because they are what the exploratory results were produced
with.

## What the experiments found

`analysis_cluster_separation.py` in the repository root reports low
silhouette scores at both the macro and sub-cluster level, and low separation
for the ground-truth labelling too. That points at the embedding space itself
not cleanly separating scar from non-scar at this granularity, which is the
context these outlier-removal experiments should be read in: Task 5's margin
histogram exists specifically to test whether IsolationForest is finding real
anomalies or just rediscovering tiles that sit on a class boundary.
