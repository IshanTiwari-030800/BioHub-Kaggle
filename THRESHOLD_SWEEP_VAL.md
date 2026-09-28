# Threshold sweep on the held-out validation set

`THRESHOLD_SWEEP.md` tuned `det_threshold`/`pool_kernel_um`/edge `threshold`
on one 15-frame slice of `6bba_372c8cb8` — which, checked here for the first
time, turns out to be a **training video**, not a held-out one. This sweep
redoes it properly: same grid logic, run across all 22 videos in the actual
validation split (`train.py`/`train_edge.py`'s `seed=0, val_frac=0.15`
default, never seen by either trained model), first 15 frames each, scored
with `repo/metrics.py`. `repo/sweep_threshold_val.py`, raw per-video output
in `weights/predictions/threshold_sweep_val_per_video.jsonl`, aggregated
per-config output in `weights/predictions/threshold_sweep_val_raw.json`.

2 of 22 videos were excluded outright: `44b6_90724892` and `6bba_767a1e17`
have zero GT-annotated nodes in their first 15 frames (a `metrics.py`
node-matching join can't score against an empty GT graph). **20 videos**,
24 configs (`det_threshold` ∈ {0.6, 0.7, 0.8} × `pool_kernel_um` ∈ {8, 10,
12, 14} × edge `threshold` ∈ {0.5, 0.7}) actually scored.

## The result is compromised by a sweep-script guard — read this before the table

`sweep_threshold.py` (the single-video script this reuses) has a
`MAX_NODES_GUARD = 4000`: if a config's running detected-node count exceeds
4000, it aborts that (video, config) pair rather than pay for ILP/attention
on a candidate set that size. **This guard exists only in the sweep script —
`predict.py`/`submission.py` have no such cap and would just run slower.**
But it means this sweep silently could not score some (video, config)
combinations at all, and `metrics.summarise()` drops NaN rows rather than
penalizing them — so a config's aggregate score is averaged only over
whichever videos happened to survive.

Checked directly: **the 8 densest videos (`n_total` — prorated
`estimated_number_of_nodes` — from 6935 to 11203) hit this guard at every
`pool_kernel_um` ≤ 12**, at every `det_threshold`. Only `pool_kernel_um=14`
partly escapes it, and even then 4 of the top-4-densest videos (all with
`n_total` > 8959) still abort. **No config in this sweep has been scored on
any video denser than ~6935 estimated cells/15 frames.**

That is not a small caveat. Confirmed by re-scoring the *same* 12 videos
(the ones that survive at `pool_kernel_um=10`) under `pool_kernel_um=14`:

| config | n videos | adj_edge_jaccard |
|---|---|---|
| det=0.8, pool=10, thr=0.5 | 12 (shared set) | **0.7339** |
| det=0.7, pool=14, thr=0.5 | same 12 | 0.6670 |

So `pool_kernel_um=14`'s lower raw aggregate (0.6096, next table) isn't
because it's a worse NMS radius — on a fair, identical video set it's
clearly worse (0.667 vs 0.734) — it's because it's scored on a harder,
denser video mix that the narrower-pool configs never had to face at all.
**The apparent config ranking below is only valid among low/medium-density
videos; it says nothing about the 8 densest ones, and pilkwang's own
`pool_kernel_um` is a fixed physical-distance constant that doesn't scale
with local cell density — the exact failure mode this was supposed to
check for.**

## Aggregate per config (20 videos where scoreable, ranked by adj_edge_jaccard)

| det_threshold | pool_kernel_um | edge threshold | n scored / 20 | adj_edge_jaccard | edge_jaccard | node_recall |
|---|---|---|---|---|---|---|
| 0.8 | 10.0 | 0.5 | 12 | **0.7339** | 0.7419 | 0.9034 |
| 0.8 | 12.0 | 0.5 | 12 | 0.7326 | 0.7405 | 0.9034 |
| 0.7 | 10.0 | 0.5 | 12 | 0.7320 | 0.7415 | 0.9039 |
| 0.7 | 12.0 | 0.5 | 12 | 0.7269 | 0.7362 | 0.9039 |
| 0.6 | 12.0 | 0.5 | 12 | 0.7265 | 0.7370 | 0.9034 |
| 0.6 | 10.0 | 0.5 | 12 | 0.7265 | 0.7370 | 0.9034 |
| 0.8 | 8.0 | 0.5 | 11 | 0.7249 | 0.7461 | 0.9461 |
| 0.7 | 8.0 | 0.5 | 10 | 0.7230 | 0.7490 | 0.9510 |
| 0.6 | 8.0 | 0.5 | 10 | 0.7207 | 0.7494 | 0.9505 |
| 0.8 | 10.0 | 0.7 | 12 | 0.7163 | 0.7209 | 0.8748 |
| 0.8 | 12.0 | 0.7 | 12 | 0.7163 | 0.7209 | 0.8748 |
| 0.7 | 12.0 | 0.7 | 12 | 0.7093 | 0.7153 | 0.8754 |
| 0.8 | 8.0 | 0.7 | 11 | 0.7093 | 0.7241 | 0.9108 |
| 0.7 | 10.0 | 0.7 | 12 | 0.7080 | 0.7140 | 0.8754 |
| 0.6 | 12.0 | 0.7 | 12 | 0.7077 | 0.7147 | 0.8754 |
| 0.6 | 10.0 | 0.7 | 12 | 0.7052 | 0.7121 | 0.8754 |
| 0.7 | 8.0 | 0.7 | 10 | 0.7052 | 0.7240 | 0.9212 |
| 0.6 | 8.0 | 0.7 | 10 | 0.6965 | 0.7176 | 0.9190 |
| 0.7 | 14.0 | 0.5 | 16 | 0.6096 | 0.6028 | 0.7500 |
| 0.8 | 14.0 | 0.5 | 16 | 0.6074 | 0.6002 | 0.7500 |
| 0.6 | 14.0 | 0.5 | 16 | 0.6045 | 0.5980 | 0.7495 |
| 0.8 | 14.0 | 0.7 | 16 | 0.5953 | 0.5870 | 0.7357 |
| 0.7 | 14.0 | 0.7 | 16 | 0.5950 | 0.5870 | 0.7361 |
| 0.6 | 14.0 | 0.7 | 16 | 0.5938 | 0.5861 | 0.7361 |

Within `pool_kernel_um` ≤ 12, `det_threshold` 0.6-0.8 are within ~1% of each
other (consistent with `THRESHOLD_SWEEP.md`'s earlier finding — detection
threshold is a weak lever once NMS pooling does the real work). Edge
`threshold=0.5` beats `0.7` in every matched pair by ~1-2% — that dimension
isn't guard-biased (edge threshold doesn't change node count) so this part
is trustworthy on its own.

## Per-video breakdown for the top-ranked config (det=0.8, pool=10.0, thr=0.5)

| video | n_total (est. cells/15f) | status | adj_edge_jaccard |
|---|---|---|---|
| 44b6_95029e92 | 774 | scored | 0.8480 |
| 6bba_2819ca14 | 830 | scored | 0.9133 |
| 6bba_4f99ce20 | 895 | scored | 0.7576 |
| 6bba_3a1849c2 | 1051 | scored | 0.7935 |
| 6bba_907271db | 1096 | scored | 0.8115 |
| 6bba_80d12824 | 1353 | scored | 0.4534 |
| 6bba_4ffd3da3 | 2646 | scored | 0.6269 |
| 44b6_e57ff5c6 | 3047 | scored | 0.3834 |
| 6bba_283bf9f1 | 3129 | scored | 0.6636 |
| 6bba_6479435d | 3518 | **aborted (guard)** | — |
| 6bba_5c824876 | 3697 | scored | 0.8137 |
| 44b6_3bb3690f | 4791 | scored | 0.7134 |
| 44b6_c8e2a523 | 4972 | scored | 0.4985 |
| 44b6_c96cfa10 | 6935 | **aborted (guard)** | — |
| 44b6_cf2536e8 | 8635 | **aborted (guard)** | — |
| 44b6_9be80b04 | 8682 | **aborted (guard)** | — |
| 44b6_b2c44266 | 8959 | **aborted (guard)** | — |
| 44b6_144b256d | 9806 | **aborted (guard)** | — |
| 6bba_57b7cc1e | 9827 | **aborted (guard)** | — |
| 44b6_e28840c6 | 11203 | **aborted (guard)** | — |

Among the 12 videos it *could* score: **min 0.383, median 0.736, max
0.913, mean 0.690** (weighted aggregate 0.7339 — TP/FP/FN-count-weighted,
so denser-among-the-measured videos count more). That's already a wide
spread on videos this config was never tuned against, before even touching
the 8 that couldn't be measured at all.

## Comparison to THRESHOLD_SWEEP.md's pick (det=0.7, pool=12.0, thr=0.7)

| | old (1 train-set video) | new (20 val videos, guard-limited) |
|---|---|---|
| adj_edge_jaccard | 0.980 | 0.709 |
| video(s) | `6bba_372c8cb8` — in the **training split** | 12/20 val videos (8 densest un-scoreable) |

Two compounding problems with the old number: it was measured on a video
the models had already trained on, and it was one video, at one density.
0.980 was never a realistic estimate of held-out performance. The
corrected, apples-to-apples version of the old config (det=0.7, pool=12,
thr=0.5 — dropping the old thr=0.7 since 0.5 is now confirmed better) on
*validation* videos is 0.7269, close to but not the top of this grid.

## Recommendation given what's actually measured

`det_threshold=0.8, pool_kernel_um=10.0, threshold=0.5` — top of the table,
but only by ~0.1-1% over five neighboring configs, i.e. **not a
statistically meaningful winner within the measured region**. The edge
`threshold=0.5 > 0.7` finding is the one piece of this sweep not undermined
by the node-count guard.

Not changing `predict.py`/`submission.py` defaults — this was a
report-only task. Before locking in any of `pool_kernel_um`'s value for a
real submission: **rerun with `MAX_NODES_GUARD` raised (or removed) so the
8 densest validation videos actually get scored.** That is the single
biggest open question this sweep leaves — if the real Kaggle test set
contains a video anywhere near `n_total` ≈ 7000-11000/15 frames (plausible,
given the 6.8x-11.5x density spread already observed across just this
20-video sample), nothing here says whether `pool_kernel_um=10-12` is even
adequate there, let alone optimal.
