# Submission-ready pipeline

Full stage 1-4 pipeline for the Biohub Cell Tracking Kaggle competition,
using our own trained models (not pilkwang's) for stages 1-3, and a direct
port of the reference notebook's graph-calibration algorithm for stage 4.

## What was built

- **`repo/graph_calibration.py`** — stage 4 (motion relink, single-parent
  repair, single-frame gap closing with pixel-level midpoint refinement,
  strict 2-frame gap recovery, conservative division insertion, short-track
  pruning, trajectory smoothing). Ported from
  `biohub-exp073-gap-5-8-public.ipynb` cell 11 (1483 lines), documented
  algorithm-by-algorithm in `README.md` section 4. Two disclosed deviations:
  1. The DeepCenter veto gate (README section 6) is omitted entirely — it
     needs a second trained model, was disabled in the scored 0.902 run
     (`BIOHUB_USE_DEEPCENTER_VETO=0`), and README's own instrumentation
     proved it checked zero nodes/edges even when enabled. Every call site
     that would have consulted it is a no-op here, matching that behavior.
  2. `read_frame` uses this repo's `zarr.open_group(path)["0"][t]` read
     path instead of the reference's manual blosc2 chunk-decompression
     (which assumes a specific Zarr v3 sharding layout) — same data, more
     robust across zarr versions/layouts.
  All numeric constants (gap distances, division radii, motion-relink gates,
  smoothing weight, etc.) are the exact "Selected (production) calibration"
  values from README section 3.1, falling back to section 3.2 defaults for
  anything not explicitly overridden in the scored run.

- **`repo/submission.py`** — orchestrates stage 1-3 (`repo/predict.py`, using
  the threshold-sweep's winning config from `THRESHOLD_SWEEP.md`:
  `det_threshold=0.7, pool_kernel_um=12.0, threshold=0.7, use_ilp=True`) per
  video, feeds the resulting ILP-solved graph through `graph_calibration.py`,
  and writes `submission.csv` + `run_stats.csv` in the exact schema from
  README section 4.14 / `nbs/submissions/submission_exp.ipynb`. Auto-detects
  the Kaggle test directory (`/kaggle/input/competitions/<comp>/test`,
  falling back to `/kaggle/input/<comp>/test`), overridable via
  `BIOHUB_TEST_DIR` for local dry runs. `--limit` / `--max-frames` flags
  exist for exactly this kind of fast local smoke test.

## Dry-run result (local, no real test set)

`BIOHUB_TEST_DIR=data/train`, first 2 videos, first 9 frames each (this
machine is CPU-only and slow — kept small on purpose to prove the pipeline
end-to-end, not to produce a meaningful score: no test-set GT exists locally
to score against anyway).

| dataset | stage1-3 nodes/edges | final nodes/edges | wall time |
|---|---|---|---|
| 44b6_0113de3b | 1921 / 1377 | 1528 / 1347 | 185.7s |
| 44b6_0b24845f | 2754 / 1723 | 1641 / 1434 | 175.4s |

Stage 4 is doing real, substantial work, not passing data through unchanged:

| effect | 44b6_0113de3b | 44b6_0b24845f |
|---|---|---|
| motion-relink replaced raw edges | 1377 → 1443 (tight=1422, relaxed=21) | 1723 → 1709 (tight=1512, relaxed=197) |
| single-parent repair dropped | 1347 | 1565 |
| gap-close pairs bridged (synthetic nodes, pixel-refined) | 40 (40 refined) | 96 (96 refined) |
| safe divisions added / candidates | 1 / 3 | 1 / 22 |
| isolated nodes pruned | 83 | 200 |
| short-track components/nodes/edges removed | 3 / 260 / 177 | 22 / 668 / 468 |
| nodes smoothed by linefit | 1526 | 1641 |

Ran the schema sanity check (`submission.py`'s `check_schema`, mirroring
`submission_exp.ipynb` cell 22) against the output: **passed** — correct
column order, node/edge row invariants (`source_id=target_id=-1` for nodes,
`node_id=t=z=y=x=-1` for edges), no dangling edges, 5950 total rows across 2
datasets (3169 node rows, 2781 edge rows).

## Caveats — read before trusting this for a real submission

1. **No score has been measured, and none is claimed.** There is no local
   test-set ground truth to check against; this dry run only proves the
   pipeline runs end-to-end and produces schema-valid output.
2. **Stage 4's numeric constants were tuned for pilkwang's detector's output
   distribution, not ours.** Gap distances, division radii, motion-relink
   gates, etc. all came from the reference notebook's own hand-tuning
   against a different upstream node/edge distribution (his T=2 dense
   backbone + `SimpleNodeTransformer`, at `det_threshold=0.97`). Our stage
   1-3 (T=3 deformable attention, `NodeTransformer`, `det_threshold=0.7`)
   produces a measurably different node density and edge-probability
   distribution (see `THRESHOLD_SWEEP.md`, `SCORING_ANALYSIS.md`). These
   constants are the correct, principled *starting point* (per the explicit
   instruction to replicate pilkwang except for model architecture), not a
   re-validated optimum for our pipeline. If a real submission underperforms,
   re-tuning stage 4 against our own output distribution (the same way
   `THRESHOLD_SWEEP.md` tuned stage 1-3) is the next lever, and is out of
   scope for this task.
3. **Needs the actual Kaggle test set to run for real.** Point it there via
   the auto-detected `/kaggle/input/competitions/biohub-cell-tracking-during-development/test`
   path, or override with `BIOHUB_TEST_DIR` if the mount differs.
4. **CPU inference is slow.** ~20-25s/window with TTA in this environment;
   a real competition test set will need either a GPU runtime or a long
   CPU budget. `--limit`/`--max-frames` are for local debugging only — don't
   use them for the real submission run.
5. Minor cosmetic note: `graph_calibration.py`'s stats dict only contains
   keys that were actually incremented at least once (a plain
   `defaultdict(int)`), unlike the reference's explicitly pre-populated
   all-zero stats dict — a counter that never fired for a given video is
   simply absent from that row's `run_stats.csv` columns (shows as blank
   when concatenated with other videos), not an explicit 0. Doesn't affect
   correctness, only `run_stats.csv` readability.
