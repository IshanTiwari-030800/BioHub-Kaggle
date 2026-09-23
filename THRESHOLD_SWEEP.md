# Detection / edge-filter threshold sweep

Sweeps `det_threshold`, `pool_kernel_um`, and edge `threshold` for
`repo/predict.py`'s pipeline, scored with the corrected metric from
`repo/metrics.py` (see `SCORING_ANALYSIS.md` for why "corrected" matters:
`n_total` must come from the GEFF's `estimated_number_of_nodes` metadata,
prorated to the frame range scored, not the raw annotated node count).
Sample: first 15 frames of `data/train/6bba_372c8cb8`, `n_total=1053.15`
(prorated from `estimated_number_of_nodes=7021` over 100 frames). ILP
enabled throughout, weights unchanged (`edge=-1.0, appearance=0.1,
disappearance=0.1, division=1.0`). Backbone encoding (the expensive part —
deformable-attention TemporalUNet3D + flip-TTA) was cached once across the
13 sliding windows and reused for every config, so the whole 19-config sweep
ran in under a minute of actual scoring time on top of the one-time cache
build.

Raw results: `weights/predictions/threshold_sweep_raw.json` (all 19 configs).

## Stage 1 — det_threshold (pool_kernel_um=3.0, threshold=0.5)

| det_threshold | n_pred_nodes | node_recall | edge_jaccard | total_node_ratio | adj_edge_jaccard |
|---|---|---|---|---|---|
| 0.5 | 1774 | 1.0 | 0.892 | +0.614 | 0.837 |
| 0.7 | 1623 | — | 0.926 | +0.484 | 0.882 |
| 0.9 | 1369 | — | 0.893 | +0.256 | 0.870 |
| **0.99 (old default)** | **1028** | **1.0** | **0.860** | **-0.040** | **0.864** |
| 0.995 | 946 | — | 0.879 | -0.112 | 0.889 |
| 0.999 | 662 | — | 0.676 | -0.383 | 0.702 |
| 0.9999 | 80 | — | 0.351 | -0.928 | 0.383 |
| 0.99999 | 2 | — | 0.008 | -0.998 | 0.009 |

The detector's sigmoid outputs are not well calibrated at the extreme tail:
past ~0.999 almost every real cell gets thrown away (2 nodes survive at
0.99999 for a scene with ~1053 real cells). The sweet spot is much lower
than the old default of 0.99 — 0.7-0.9 already beats it, driven by
`edge_jaccard` recovering as more real cells survive, while `total_node_ratio`
stays in a moderate positive range rather than swinging deeply negative
(missed real cells) or wildly positive (duplicate spam).

## Stage 2 — pool_kernel_um at det_threshold=0.7 and 0.995

| det_threshold | pool_kernel_um | n_pred_nodes | edge_jaccard | total_node_ratio | adj_edge_jaccard |
|---|---|---|---|---|---|
| 0.7 | 5.0 | 1623 | 0.926 | +0.484 | 0.882 |
| 0.7 | 8.0 | 1473 | 1.000 | +0.360 | 0.964 |
| 0.7 | 12.0 | 1299 | 1.000 | +0.197 | **0.980** |
| 0.7 | 15.0 | 1091 | 0.934 | +0.013 | 0.933 |
| 0.995 | 5.0 | 946 | 0.879 | -0.112 | 0.889 |
| 0.995 | 8.0 | 924 | 0.934 | -0.128 | 0.946 |
| 0.995 | 12.0 | 906 | 0.941 | -0.145 | 0.954 |
| 0.995 | 15.0 | 837 | 0.941 | -0.211 | 0.961 |

Widening the NMS suppression radius (`pool_kernel_um`, default was 3.0)
matters *more* than raising `det_threshold` for killing near-duplicate peaks
around the same real cell — at `det_threshold=0.7`, going from 3um to 12um
takes `edge_jaccard` from 0.86 to a clean 1.000 (zero false-positive edges)
while node count drops from 1774 to 1299. Past 12um it starts cutting into
real, distinct cells (edge_jaccard drops back to 0.934 at 15um) — 12um is
the local optimum on this sample.

## Stage 3 — edge threshold at (det_threshold=0.7, pool_kernel_um=12.0)

| edge threshold | n_final_edges | edge_tp | edge_fp | edge_fn | edge_jaccard | adj_edge_jaccard |
|---|---|---|---|---|---|---|
| 0.3 | 1164 | 130 | 0 | 0 | 1.000 | 0.979 |
| 0.5 | — | — | — | — | 1.000 | 0.980 |
| **0.7** | 1138 | 130 | 0 | 0 | 1.000 | **0.980** |
| 0.9 | 1006 | 125 | 1 | 5 | 0.954 | 0.943 |

Edge threshold barely matters once detection/NMS are dialed in (0.3-0.7 are
all within noise of each other, all with **zero** edge FP/FN) — 0.9 starts
losing real edges. Not a meaningful lever; leave at 0.5-0.7.

## Recommendation

```
det_threshold   = 0.7    (was 0.99)
pool_kernel_um  = 12.0   (was 3.0)
threshold       = 0.7    (was 0.5, but 0.3-0.7 are equivalent)
edge_activation = softmax
use_ilp         = True
ilp_edge_weight = -1.0, ilp_appearance_weight = 0.1,
ilp_disappearance_weight = 0.1, ilp_division_weight = 1.0   (unchanged)
```

`adj_edge_jaccard = 0.980` vs. `0.864` at the old defaults — driven almost
entirely by `edge_jaccard` going from 0.860 (13 FP / 7 FN edges) to a clean
1.000 (0 FP / 0 FN edges) once `pool_kernel_um` is widened enough to stop
counting the same cell twice. This was only measurable at all because of the
`n_total` fix in `SCORING_ANALYSIS.md` — under the old (wrong) metric,
`det_threshold` looked like the dominant lever and `pool_kernel_um` wasn't
even a suspect.

Caveat: tuned on one 15-frame slice of one video. Before locking this in for
a real submission, spot-check on 2-3 more videos with different cell
densities — `pool_kernel_um=12.0` in particular is a physical-distance
constant that could be too aggressive on a video with genuinely
closely-packed cells.
