# Why is the raw score so high — and is it gameable?

Investigated whether `node_recall=1.0` / `edge_jaccard=0.860` (from the earlier
`PIPELINE_VERIFICATION.md` run: 1011 predicted nodes vs. 143 GT nodes on a
15-frame slice of `6bba_372c8cb8`) are inflated by an evaluation-matching
flaw, or genuinely reflect good performance. Reproducible via
`repo/scoring_analysis.py`.

## Part 1 — is the matching gameable by spamming detections?

**No.** `tracksdata`'s `.match()` (`_base_graph.py:1349`) defaults every
`Matching` strategy to `optimal=True`. With that flag, `_match_single_frame`
(`tracksdata/metrics/_ctc_metrics.py:36-113`) runs a genuine per-timepoint
**one-to-one bipartite assignment**
(`scipy.sparse.csgraph.min_weight_full_bipartite_matching`, falling back to
`scipy.optimize.linear_sum_assignment`) — every predicted node can claim at
most one GT node and vice versa. Spamming ten detections next to one real
cell does not give ten true positives; it gives (at best) one, with the rest
competing for other GT nodes or going unmatched. `pilkwang`'s own
`_evaluate()` call (`metrics.py:162-163`, vendored into this repo) builds
`DistanceMatching(max_distance=..., scale=...)` with the same default
`optimal=True`. So `node_recall=1.0` and `edge_jaccard=0.860` are legitimate:
among the 143 GT-annotated cells, the detector found a close match for every
one of them, and the edge scorer + ILP link 86% of the real edges correctly
among those matched cells.

**Distance-bucketed diagnostic** (nearest-GT distance for every one of the
1011 predicted nodes, `max_distance=7um`):

| Bucket | Count | % | Median mean intensity |
|---|---|---|---|
| ≤7um (match candidates) | 145 | 14.3% | 517.0 |
| 7-14um (near-duplicate zone) | 37 | 3.7% | 327.1 |
| 14-28um | 416 | 41.1% | 485.7 |
| >28um (background zone) | 413 | 40.9% | 454.9 |
| GT cells (reference) | 143 | — | 501.4 |
| Random background voxels | 200 | — | 61.3 |

This rules out the "spam near-duplicates" hypothesis directly: only 3.7% of
predictions land in the 7-14um near-duplicate zone, and those are the one
bucket that's actually *dim* relative to true cells (327 vs. 501-517) —
consistent with weak, partially-suppressed duplicate blobs, a real but small
effect (candidate `pool_kernel_um` tuning target, not urgent).

The much bigger story is the 82% of predictions sitting 14-28+um from any
GT node: their intensity (455-486) is **far closer to a real cell (501) than
to background noise (61)**. These are not hallucinated noise — they are
landing on genuinely bright, cell-shaped structures. Given the ground truth
here is known to be a *sparse* subset of real cells (see Part 2), the most
likely explanation is that most of this "82%" is real, unannotated cells,
not detector error.

## Part 2 — the actual scoring bug: wrong `n_total`

`PIPELINE_VERIFICATION.md`'s adjusted score (`adj_edge_jaccard=0.338`) used
`n_total = gt.num_nodes()` (143, the raw annotated node count in the
15-frame window) as the "true" cell count for
`total_node_ratio = (num_pred - n_total) / n_total`. That is **not** what
pilkwang's own pipeline does. His `scripts/evaluate.py:41-51`
(`_read_estimated_n_total`) explicitly reads `estimated_number_of_nodes`
from the GEFF's metadata `extra` dict instead of using the annotated node
count — because, per that function's own docstring context, GT annotation
here is known to be sparse.

Checked this dataset's actual metadata:

```
GeffMetadata.read('data/train/6bba_372c8cb8.geff').extra
  -> {'estimated_number_of_nodes': 7021}
```

7021 estimated true node-instances across the full 100-frame video vs. 1036
raw annotated nodes — a **6.8x** gap between "real cell count" and
"annotated cell count" for this video, confirming the sparsity pilkwang's
code is designed around, independent of anything this pipeline does.

Prorating to the 15-frame window (`7021 * 15/100 = 1053.1`) and recomputing:

| `n_total` used | total_node_ratio | adj_edge_jaccard |
|---|---|---|
| raw annotated GT (143) — **what `PIPELINE_VERIFICATION.md` used** | +6.070 | 0.338 |
| `estimated_number_of_nodes`, prorated (1053.1) — **what pilkwang's own code uses** | -0.040 | **0.864** |

1011 predicted nodes against an estimated true count of ~1053 is a ratio of
0.96 — i.e. **not overpredicting**, essentially matching the real cell
count. The earlier "0.338, crushed by 7x overprediction" conclusion was an
artifact of comparing against the wrong denominator (sparse annotations
instead of the estimated true count), not a real detector flaw.
`adj_edge_jaccard≈0.86` is the number that reflects the same formula
pilkwang's pipeline actually scores with.

## Recommendation

1. **Don't do hard-negative mining on "far from GT" detections.** The
   intensity evidence says most of them are real, unannotated cells; training
   the detector to suppress them would actively teach it to miss real cells
   for no metric benefit, since the real scoring formula doesn't penalize
   them once `n_total` is computed correctly.
2. **Fix `n_total` wherever this repo scores itself going forward.** Any
   future evaluation script (including a full submission dry-run) must read
   `estimated_number_of_nodes` from each video's GEFF metadata and prorate it
   to whatever frame range was scored — not fall back to
   `gt_graph.num_nodes()`. `repo/scoring_analysis.py` does this correctly and
   can be reused/extended for that purpose.
3. **`det_threshold` sweeping should target the corrected metric.** A
   threshold sweep that uses raw annotated GT counts as ground truth for
   "how many nodes should exist" will systematically push the threshold too
   high (chasing an artificially low overprediction penalty that isn't real).
   Whoever runs the threshold sweep should use `estimated_number_of_nodes`
   (prorated per-video), not annotated node counts, when judging
   over/under-prediction.
4. The one legitimate, small hard-negative-mining target is the 3.7%
   near-duplicate bucket (7-14um, dimmer than real cells) — a `pool_kernel_um`
   tuning question, not worth a training-loss change at this scale.
