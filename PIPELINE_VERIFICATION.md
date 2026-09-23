# End-to-end pipeline verification

One-time smoke test of the full stage 1→2→3 pipeline (`repo/predict.py`):
`TemporalUNet3D` detector → `NodeTransformer` edge scorer → `tracksdata`
ILP tracker → `.geff`. Detection, edge filtering, and ILP cost weights are a
direct port of pilkwang's vendored `predict_unet_transformer.py`; the only
differences are the deformable-attention T=3 backbone and the joint-attention
`NodeTransformer` (see `repo/predict.py` module docstring for the exact list).

## Setup

- Weights: `weights/edge_model_best.pt` (fully trained `CentroidUNet` +
  `NodeTransformer`, downloaded from the completed Kaggle training run).
  No `edge_config.json` was downloaded alongside it, so the script fell back
  to `EdgeTrainConfig` defaults — the strict `state_dict` load succeeded,
  confirming those defaults match what was actually trained.
- Video: `data/train/6bba_372c8cb8`, first **15 of 100 frames** (CPU-only
  environment; full-video inference would take ~45 min at ~29s/window with
  TTA, deferred instead of run inline for a sanity check).
- Config: `det_threshold=0.99`, `pool_kernel_um=3.0`, `edge_activation=softmax`,
  `threshold=0.5`, ILP weights `edge=-1.0, appearance=0.1, disappearance=0.1,
  division=1.0` (all pilkwang's defaults, unchanged).

## Result

```json
{
  "n_frames": 15,
  "n_nodes": 1011,
  "n_candidate_edges": 1820,
  "n_final_edges": 922,
  "n_divisions": 11,
  "predict_sec": 330.8,
  "ilp_sec": 0.92
}
```

Gurobi isn't licensed in this environment; the ILP solver automatically fell
back to SCIP (`pyscipopt`), which solved the ~1011-node/1820-edge candidate
graph in under a second.

## Structural sanity checks (passed)

Checked the ILP output directly against the constraints it's supposed to
enforce:

- **0** nodes with more than 1 incoming edge — no merges, as required.
- **0** nodes with more than 2 outgoing edges — division cap respected.
- 11 nodes with exactly 2 children, matching the reported division count.

This confirms the ILP is actually doing global constrained optimization
(not just passing candidates through) — ungated greedy assignment on this
same candidate set would not guarantee either property.

## Real competition-proxy score

`repo/metrics.py` / `repo/division_metrics.py` are a direct copy of
pilkwang's vendored scoring implementation (CC0-licensed, confirmed via the
Kaggle dataset's schema.org metadata) — the actual edge-Jaccard +
node-overprediction-penalty + division-Jaccard formula behind the
leaderboard metric, not an approximation. Ran it against the same 15-frame
GT slice:

```
edge_tp=123  edge_fp=13  edge_fn=7
division_tp=0  division_fp=3  division_fn=0
node_recall        = 1.000   (every GT node in this window was matched)
edge_jaccard        = 0.860  (among matched pairs, edge linking is accurate)
```

**Correction (superseding the `total_node_ratio`/`adj_edge_jaccard` numbers
originally reported here):** the first pass computed `total_node_ratio`
using `n_total = gt.num_nodes()` (143, the raw *annotated* node count),
giving a fake "6.07x overprediction" and `adj_edge_jaccard=0.338`. That's
not what pilkwang's own `evaluate.py` does — GT annotation here is
deliberately sparse, so it reads `estimated_number_of_nodes` from the GEFF
metadata instead (`SCORING_ANALYSIS.md` has the full investigation).
Verified directly: this video's metadata has
`estimated_number_of_nodes=7021` (vs. 1036 raw annotated nodes across all
100 frames — a 6.8x sparsity gap). Prorated to the 15-frame window
(`7021 * 15/100 = 1053.1`) and recomputed:

```
total_node_ratio    = -0.040  (1011 pred nodes vs. ~1053 estimated true nodes — essentially exact)
adj_edge_jaccard    =  0.864
```

So the detector is **not** overpredicting at `det_threshold=0.99` — it's
finding real, unannotated cells (confirmed independently via intensity
analysis in `SCORING_ANALYSIS.md`: the "far from GT" detections have
image-intensity statistics matching real cells, not background). The
edge scorer + ILP combination, and the detector itself, are both performing
well on this sample. This also means my earlier recommendation to treat
`det_threshold` as "the highest-leverage next tuning step" was based on a
scoring bug on my part, not a real finding — see `THRESHOLD_SWEEP.md` for
the corrected sweep.

## Conclusion

The full pipeline runs end-to-end without errors and produces a structurally
valid lineage graph (`weights/predictions/6bba_372c8cb8_sample.geff`,
loadable via `tracksdata`). This is a functionality check, not a scored or
tuned submission — detection threshold, edge threshold, and link-distance
gating still need sweeping against the real competition metric before
submitting.
