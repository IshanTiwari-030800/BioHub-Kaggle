# Stage-4 calibration sweep

Validates the caveat in `SUBMISSION_PIPELINE.md`: `repo/graph_calibration.py`'s
constants were copied verbatim from pilkwang's README-documented calibration,
tuned against *his* stage 1-3 output (T=2 backbone, `det_threshold=0.97`), not
ours (T=3 deformable attention, `det_threshold=0.7`, `pool_kernel_um=12.0` —
`THRESHOLD_SWEEP.md`'s winning config). This sweeps those constants against
our own output, scored with `repo/metrics.py`'s real metric.

Sample: `data/train/6bba_372c8cb8`, first 15 frames, `n_total=1053.15`
(prorated `estimated_number_of_nodes`, per `SCORING_ANALYSIS.md`). Stage 1-3
ran once (359s, producing 1261 nodes / 1138 edges after ILP) and stayed fixed
for the whole sweep — stage 4 is pure graph post-processing, no neural net, so
each of the 39 configs below only costs the calibration + scoring time.
Raw results: `weights/predictions/stage4_sweep_raw.json`.

## A real bug found and fixed along the way

First sweep attempt gave internally inconsistent results — the same config,
called twice with different unrelated overrides in between, produced
different scores. Root cause: `calibrate_graph` mutates its `nodes_by_id`
input dict in place (e.g. `close_single_frame_gaps` does
`nodes_by_id[middle_id] = {...}` directly into the caller's dict, no copy).
That's harmless for `submission.py`'s single call per video, but fatal for a
sweep that calls it repeatedly on what's supposed to be the same base graph —
each call was silently building on the *previous* call's synthetic
gap-closed/pruned state. Fixed by deep-copying `nodes_by_id`/`raw_edges`
before every call (`repo/sweep_stage4.py`); re-ran the entire sweep after the
fix. All numbers below are post-fix.

## A metric quirk worth knowing about before trusting any of this

`adj_edge_jaccard = max(0, edge_jaccard * (1 - 0.1 * total_node_ratio))` can
exceed **1.0** whenever `total_node_ratio` goes negative (underpredicting
relative to the estimated true count) — the "penalty" term becomes a bonus.
Several configs below hit 1.00-1.03 this way (e.g. `output_min_track_len=12`
→ 1.0294) purely by deleting more nodes, with `node_recall` staying at a flat
1.000 the entire time (2 through 12) because none of the 143 GT-annotated
nodes in this window happen to live in a short track. That's a property of
*this specific, short, well-linked sample* — aggressively deleting short
tracks in general would delete real, correctly-tracked cells on a video where
GT nodes aren't so conveniently clustered into long tracks. Treat any
"winner" that relies on pushing `total_node_ratio` negative as *unvalidated
on this evidence*, not a real improvement — flagged per-stage below.

## Stage A — motion relink

| config | n_nodes | recall | edge_j | ratio | adj_j |
|---|---|---|---|---|---|
| baseline (bonus=1.0, tight=6.0, relaxed=10.0) | 1094 | 1.000 | 1.000 | +0.039 | 0.9961 |
| motion_relink **off** | 1108 | 1.000 | 1.000 | +0.052 | 0.9948 |
| learned_bonus ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0} | 1094 (all) | 1.000 | 1.000 | +0.039 | 0.9961 (all identical) |
| tight=4.0, relaxed=8.0 | 1073 | 1.000 | 1.000 | +0.019 | **0.9981** |
| tight=8.0, relaxed=12.0 | 1100 | 1.000 | 1.000 | +0.044 | 0.9956 |
| tight=10.0, relaxed=15.0 | 1127 | 1.000 | 0.977 | +0.070 | 0.9704 |
| tight=12.0, relaxed=18.0 | 1136 | 1.000 | 0.955 | +0.079 | 0.9477 |
| velocity_weight ∈ {0.0..1.0} | 1094 (all) | 1.000 | 1.000 | +0.039 | 0.9961 (all identical) |

**`motion_relink_learned_bonus` and `motion_relink_velocity_weight` had zero
measurable effect** across their full tested range — every value produced
byte-identical node/edge counts. At this node density, motion + raw distance
apparently dominate the Hungarian cost matrix enough that neither term ever
flips an assignment; not investigated further (would need to inspect the
actual cost matrices to confirm vs. rule out a wiring issue).

**Winner: `tight_um=4.0, relaxed_um=8.0`** (adj_j 0.9961→0.9981). This is a
*clean* improvement, not the negative-ratio quirk — `total_node_ratio` stays
positive (+0.019, better-calibrated, not flipped negative) and
`edge_jaccard`/`node_recall` are both still a perfect 1.000. Tighter gates
mean fewer speculative long-range relink matches, consistent with our
already-clean ILP output not needing a wide relink net. **Applied to
`repo/graph_calibration.py`'s defaults.**

## Stage B — gap closing (at the stage-A winner)

| config | n_nodes | recall | edge_j | ratio | adj_j |
|---|---|---|---|---|---|
| gap_close_um=3.0 | 1059 | 1.000 | 1.000 | +0.006 | 0.9994 |
| gap_close_um=4.4 | 1073 | 1.000 | 1.000 | +0.019 | 0.9981 |
| gap_close_um=5.8 (baseline) | 1073 | 1.000 | 1.000 | +0.019 | 0.9981 |
| gap_close_um=7.0 | 1095 | 1.000 | 1.000 | +0.040 | 0.9960 |
| gap_close_um=9.0 | 1124 | 1.000 | 1.000 | +0.067 | 0.9933 |
| gap_close_um=11.6 | 1155 | 1.000 | 1.000 | +0.097 | 0.9903 |
| gap_close **off** | 1003 | 1.000 | 1.000 | -0.048 | 1.0048 |

"gap_close off" numerically wins, but by pushing `total_node_ratio` negative
— the quirk above. At `det_threshold=0.7` our pipeline already detects most
real cells directly, so there may genuinely be fewer real single-frame gaps
left to bridge than in pilkwang's higher-threshold regime — but this sample
can't distinguish "gap-closing is genuinely less useful now" from "gap-closing
adds a few extra nodes that happen to cost more than they're worth under this
formula's sign convention." **Not changed — left at the pilkwang default
(5.8um)** pending a second sample or real submission feedback.

## Stage C — safe division radii (not trustworthy on this sample)

`baseline` division counts: `tp=0, fn=0` — **zero real GT divisions** exist
in this 15-frame window. Every radius tested (3.0-8.0um) and division on/off
produced identical or near-identical scores because there's nothing here for
division logic to get right or wrong. **No conclusion possible; not
changed.** Would need a video with actual GT divisions (the original
`PIPELINE_VERIFICATION.md` run found 11 predicted divisions but that was
before the `n_total` fix and on a different config — a real check needs a
fresh run on a division-containing video, out of scope here given time spent
debugging the mutation bug above).

## Stage D — short-track length / linefit smoothing (at stage-B winner, gap-close off)

| config | n_nodes | recall | edge_j | ratio | adj_j |
|---|---|---|---|---|---|
| min_track_len=2 | 1234 | 1.000 | 1.000 | +0.172 | 0.9828 |
| min_track_len=4 | 1136 | 1.000 | 1.000 | +0.079 | 0.9921 |
| min_track_len=6 (baseline) | 1003 | 1.000 | 1.000 | -0.048 | 1.0048 |
| min_track_len=8 | 905 | 1.000 | 1.000 | -0.141 | 1.0141 |
| min_track_len=12 | 744 | 1.000 | 1.000 | -0.294 | 1.0294 |
| linefit_weight ∈ {0.0, 0.4, 0.8, 1.0} | 1003 (all) | 1.000 | 1.000 | -0.048 | 1.0048 (all identical) |

Monotonically "better" as `min_track_len` increases — textbook negative-ratio
gaming, `node_recall` never moves off 1.000 while nodes keep dropping.
**Not changed.** `linefit_weight` (trajectory smoothing) has zero effect on
any of these integer-valued metrics by construction — it only nudges
coordinates, which neither `edge_jaccard` nor `node_recall`/`total_node_ratio`
are sensitive to at this sample's distance-matching tolerance (7um).

## What actually changed

`repo/graph_calibration.py`: `motion_relink_tight_um` 6.0→**4.0**,
`motion_relink_relaxed_um` 10.0→**8.0**. Everything else (learned_bonus,
gap-close distance, division radii, min track length, linefit weight) is
**unchanged from the pilkwang-copied defaults** — either no measurable effect
was found, or the only "improvement" available relied on the
`total_node_ratio` sign-flip quirk rather than a real, generalizable quality
gain.

## Caveats

1. One 15-frame sample, zero real divisions in it — insufficient to validate
   gap-close/division/short-track changes either way. The motion-relink gate
   change is the only one with clean, non-gamed supporting evidence.
2. No real submission score exists to confirm any of this end-to-end.
3. Did not get to a second, busier video with real divisions (flagged as
   optional in scope, deprioritized after the mutation-bug detour consumed
   the time budget) — recommended before trusting the gap-close/short-track
   findings either direction.
