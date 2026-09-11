# Biohub Cell Tracking — Algorithm Notes (0.902 public LB)

This document explains, in full detail, how `biohub-exp073-gap-5-8-public.ipynb` turns raw 3D
microscopy volumes into a submission for the **Biohub Cell Tracking During Development** Kaggle
competition, and scored **0.902** on the public leaderboard.

The competition task: given 3D+time microscopy volumes (`.zarr`), output every detected nucleus
per frame ("node") and every parent→child association across consecutive frames ("edge"),
including cell divisions (one parent → two children).

> **Score provenance** (from the notebook): submission `54633411` scored `0.902`;
> its `submission.csv` SHA-256 is `8c73d776abca799a37c2bd24768a1bbb510418c3327192430730540d56cea698`.

---

## 1. Overview

The pipeline has four stages:

```
 raw .zarr volumes
        │
        ▼
 (1) TemporalUNet3D detection  ──► per-frame 3D nucleus centroids
        │
        ▼
 (2) Node-transformer edge scoring ──► learned probability that node A (frame t) links to node B (frame t+1)
        │
        ▼
 (3) ILP assignment  ──► one raw candidate lineage graph per video, written as .geff
        │
        ▼
 (4) Graph calibration (THIS NOTEBOOK'S CODE) ──► submission.csv
        - motion-based relinking (Hungarian assignment)
        - single-frame gap closing (with pixel-level midpoint refinement)
        - conservative division insertion
        - short-track pruning
        - trajectory smoothing
```

**The critical insight the notebook states explicitly:** the model weights are *unchanged* from
an earlier, lower-scoring run. The 0.902 result comes almost entirely from **stage 4** — a
deterministic, hand-tuned graph post-processing/calibration layer applied to the raw ILP output.
Stage 4 is fully implemented in the notebook and is the part you can copy verbatim. Stages 1–3
call out to a pretrained model + script that ships in a separate, private Kaggle dataset.

---

## 2. What's reproducible vs. what's external

This matters a lot if your goal is to recreate the solution.

| Stage | Code present in notebook? | What you'd need |
|---|---|---|
| (1) TemporalUNet3D detector | **No.** Only invoked via CLI. | The trained model weights + `scripts/predict_unet_transformer.py`, both shipped in the Kaggle dataset `pilkwang/biohub-tracking-support-pack-50ep-v1`. Not visible in this notebook. |
| (2) Node-transformer edge scorer | **No.** Same as above — internal to the vendored script. | Same support-pack dataset. |
| (3) ILP tracker | **No.** Invoked via `--use-ilp` CLI flag on the same script, using the `tracksdata`/`ilpy` libraries. | Same support-pack dataset (the actual ILP formulation/costs are wired up inside the vendored script, not shown here). |
| (3.5) Detection TTA patch | **Yes**, in full. | A source-patching snippet (see §4.3) that upgrades 4-way to 8-way test-time augmentation by string-replacing a block inside the vendored script before it runs. |
| (4) Graph calibration (motion relink, gap-close, safe divisions, short-track filter, smoothing) | **Yes, completely.** ~1,480 lines, all in one cell. | Nothing external — this only needs the `.geff` graph tracksdata produced in stage 3 plus the raw `.zarr` volumes (for pixel-level gap-midpoint refinement). This is copy-paste reproducible. |
| (5) Dormant "DeepCenter" veto gate | **Yes**, full model + gating code, but **disabled** in the scored config (`BIOHUB_USE_DEEPCENTER_VETO=0`). Diagnostics proved it checked zero nodes/edges in the scored run, so it's dead weight — documented here for completeness but you can skip it entirely. | N/A (inert) |

If you want to recreate the *entire* 0.902 pipeline, you need the support-pack weights/script for
stages 1–3. If you want to recreate the *graph calibration algorithm* (stage 4, the part that
actually drives the score improvement), everything you need is below.

---

## 3. Configuration layer (cells 0 & 4/5)

Every tunable parameter is an environment variable read with `os.environ.get(NAME, default)`, so
the whole calibration is one big config object. The notebook sets ~30 of these explicitly to lock
in the "selected calibration" (preset name `exp073_public0902_gap58`); everything not explicitly
set falls back to the default embedded in the `os.environ.get(...)` call in cell 5.

### 3.1 Selected (production) calibration

| Setting | Env var | Value | Purpose |
|---|---|---:|---|
| Detection threshold | `BIOHUB_DET_THRESHOLD` | `0.970` | Confidence cutoff for the detector; keeps useful candidates, drops the noisy tail. |
| Motion-relink learned-edge bonus | `BIOHUB_MOTION_RELINK_LEARNED_BONUS` | `1.0` | Weight given to the node-transformer's edge probability inside the motion-relink cost function (§4.5). Higher = trust the learned model more over pure motion. |
| Gap-close max gap | `BIOHUB_GAP_CLOSE_MAX_GAP` | `2` (but effectively clamped to `1`, see §4.6) | How many missing frames a single gap-closing pass will bridge. |
| Gap-close distance | `BIOHUB_GAP_CLOSE_UM` | `5.8 µm` | Max physical distance (scaled by gap size) allowed between a track's end and a track's start to bridge them. |
| Minimum track length | `BIOHUB_OUTPUT_MIN_TRACK_LEN` | `6` nodes | Connected components (lineages) shorter than this are deleted, unless they contain a division. |
| Keep division components | `BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS` | `1` (true) | Components containing a division are never removed by the short-track filter, regardless of length. |
| Safe-division parent radius | `BIOHUB_SAFE_DIV_MAX_UM` | `4.66 µm` | Max parent→new-daughter distance allowed when adding a division. |
| Safe-division sister radius | `BIOHUB_SAFE_DIV_SISTER_MAX_UM` | `8.5 µm` | Max distance between the two daughters allowed when adding a division. |
| Safe-division existing-child radius | `BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM` | `7.65 µm` | The parent's *existing* single child link must already be this tight before a second child can be proposed. |
| Safe-division per-frame cap | `BIOHUB_SAFE_DIV_FRAME_FRAC_CAP` | `0.76%` of that frame's source count | Caps how many new divisions can be added in a single frame. |
| Safe-division global cap | `BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP` | `0.375%` of all edges | Caps total new divisions added across the whole video. |
| 2-frame gap recovery | `BIOHUB_OUTPUT_GAP2_RECOVERY` | `0` (off) | Stricter, separate pass for exactly-2-missing-frame gaps; disabled so a loose gap-close override can't accidentally create non-consecutive edges (see §4.6/§4.7). |
| Adaptive short-track rescue | `BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE` | `0` (off) | Would re-admit some short tracks if too many nodes got removed; disabled in this run. |
| DeepCenter veto gate | `BIOHUB_USE_DEEPCENTER_VETO` | `0` (off) | Auxiliary centerness-confidence gate on new nodes; proven inert, disabled (§6). |

### 3.2 Other defaults relevant to the algorithm (from cell 5, not overridden)

| Env var | Default | Purpose |
|---|---:|---|
| `BIOHUB_OUTPUT_EDGE_MAX_UM` | `14.0 µm` | Hard cap on any single-frame edge length; longer edges are dropped outright. |
| `BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME` | `1` | Drop any edge that doesn't connect exactly consecutive frames. |
| `BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR` | `1` | Keep only the best incoming edge per node (no node may have 2 parents). |
| `BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR` | `0` | (Off) Would similarly cap outgoing edges to 1 per node — divisions rely on this staying off. |
| `BIOHUB_OUTPUT_PRUNE_ISOLATED` | `1` | Drop nodes with no edges at all after repair. |
| `BIOHUB_OUTPUT_MOTION_RELINK` | `1` | Enables the motion-relink pass (§4.5), which **replaces** the raw ILP edges. |
| `BIOHUB_MOTION_RELINK_TIGHT_UM` / `_RELAXED_UM` | `6.0` / `10.0 µm` | Two-pass assignment gates (§4.5). |
| `BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT` | `0.5` | Weight on the constant-velocity term of the motion prediction. |
| `BIOHUB_MOTION_RELINK_MAX_FRAME_NODES` | `2600` | Safety valve: skip motion-relink entirely (fall back to raw edges) if any frame is too crowded for the O(n²) assignment to be practical. |
| `BIOHUB_GAP_CLOSE_REUSE_EXISTING` / `_REUSE_UM` | `1` / `3.2 µm` | Prefer reusing a real, previously-untracked detection as the gap's middle node over synthesizing one. |
| `BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC` / `_ABS` | `5%` / `2000` | Caps on synthetic nodes inserted by gap-closing. |
| `BIOHUB_GAP_REFINE_SYNTHETIC` | `1` | Enables pixel-intensity refinement of synthetic gap midpoints (§4.6). |
| `BIOHUB_GAP_REFINE_WIN_Z` / `_WIN_YX` / `_MAX_SHIFT_UM` | `1` / `3` / `3.2 µm` | Refinement search window (voxels) and max allowed shift. |
| `BIOHUB_OUTPUT_LINEFIT_SMOOTH` / `_WEIGHT` / `_WINDOW` | `1` / `0.8` / `2` | Trajectory smoothing pass (§4.9). |
| `BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER` | `0` | (Off) Would post-hoc validate/collapse any node with >1 child using `DIV_PARENT_MAX_UM`/`DIV_SISTER_MAX_UM`; unnecessary here because `add_safe_divisions_postlink` already only *adds* geometrically-valid divisions. |

Voxel physical scale used everywhere distances are computed:

```python
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)  # (z, y, x) microns per voxel
```

All "distance in µm" values in this document are Euclidean distances after applying this
per-axis scale — i.e. `sqrt((dz*1.625)^2 + (dy*0.40625)^2 + (dx*0.40625)^2)`.

---

## 4. Stage-by-stage detail

### 4.1 Dependency & artifact setup (cell 7)

Not algorithmically interesting, but necessary for reproduction:

- Searches a fixed list of Kaggle input paths for a directory matching
  `TARGET_ARTIFACT_SLUG = "biohub-tracking-support-pack-50ep-v1"`, verified by the presence of
  `repo/` (or `repo.zip`) and `weights/unet_transformer/split_0/edge_predictor_best.pth` (or
  `weights.zip`).
- Resolves ~40 pinned Python dependencies (`tracksdata`, `zarr>=3.0.10,<4`, `pyscipopt`,
  `geff>=1.1.3.1.1`, `geff-spec<1.2`, `ilpy>=0.5.1`, `polars>=1.36`, `blosc2`, `dask`,
  `rustworkx>=0.17.1`, `sqlalchemy>=2`, etc.), preferring **offline wheels** attached to the
  Kaggle input over PyPI, and only using PyPI if `BIOHUB_ALLOW_PIP_INSTALL=1` (off in a scored,
  internet-off run).
- Symlinks (or copies, as fallback) the vendored inference repo and weights into
  `/kaggle/working/tracking_repo`.

If you're recreating this from scratch, the equivalent of this step is: get (or train) a
TemporalUNet3D detector + node-transformer edge scorer checkpoint, and a driver script that can
run inference + ILP tracking and emit `.geff` graphs, matching the CLI contract in §4.3.

### 4.2 tracksdata / geff graph model

Candidate graphs are read with:

```python
graph = td.graph.IndexedRXGraph.from_geff(path)
```

`.geff` ("graph exchange file format") is the on-disk graph format used by the `tracksdata`
library; each graph has node attributes `t, z, y, x` and edge attributes including a learned
`edge_prob`. The notebook pulls everything back into plain Python dicts for the calibration pass:

```python
nodes_by_id = {node_id: {"node_id", "t", "z", "y", "x"}, ...}
raw_edges = [{"source_id", "target_id", "edge_prob"}, ...]
```

All of stage 4 operates on these two plain structures — no graph library is required to
reimplement it, only to read the `.geff` in the first place.

### 4.3 Detection TTA patch + prediction invocation (cell 9)

**Patch.** Before running the vendored script, the notebook does an in-place string replacement
inside `scripts/predict_unet_transformer.py`, upgrading the detector's test-time augmentation from
a 4-way average (identity + 3 flips) to a full 8-way dihedral-group (D4) average:

- 3 flip augmentations: flip last axis, flip second-to-last axis, flip both (as before).
- **New:** 90° and 270° rotations (`torch.rot90(imgs, k, dims=(-2,-1))` for `k in (1,3)`),
  undone on the output the same way.
- **New:** a transpose of the last two axes (swap Y/X), undone symmetrically.
- **New:** an anti-transpose (rotate 90° then transpose), undone symmetrically.
- All 8 view predictions (1 identity implied by the surrounding code + 7 augmented) are summed and
  divided by the view count, i.e. straightforward TTA logit averaging.

The patch is applied defensively: it looks for an exact source substring and only writes back if
found, printing a warning instead of failing if the vendored script has since changed.

**Invocation.** The (patched) script is run once as a subprocess per experiment, covering all test
videos:

```bash
python scripts/predict_unet_transformer.py \
  --data-dir <TEST_DIR> \
  --splits kaggle_test_splits_50ep.json --split 0 \
  --weights weights/unet_transformer/split_0/edge_predictor_best.pth \
  --unet-batch-size 4 \
  --det-threshold 0.97 \
  --ilp-edge-weight -1.0 \
  --ilp-appearance-weight 0.1 \
  --ilp-disappearance-weight 0.1 \
  --ilp-division-weight 1.0 \
  --use-ilp
```

Output: one `.geff` graph per test video under `tracking_repo/predictions/*/unet_transformer/split_0/*.geff`.

### 4.4 Basic edge filtering (`filter_output_graph`, entry point)

For every raw edge coming out of the `.geff`:

1. Drop it if source/target node is missing.
2. If `OUTPUT_ENFORCE_NEXT_FRAME` (on): drop it unless `target.t == source.t + 1`.
3. Compute 3D physical distance (`edge_distance_um`, using `VOXEL_SCALE_UM`) and store it on the edge.
4. Drop it if that distance exceeds `OUTPUT_EDGE_MAX_UM` (14 µm).

### 4.5 Motion-based relinking (`motion_relink_edges`)

This is the single biggest structural change to the raw ILP graph: **it discards the raw ILP
edges and rebuilds every frame-to-frame association from a fresh optimal assignment**, informed by
both physics (motion) and the learned edge probabilities.

For each node, a **predicted next position** is computed by simple constant-velocity extrapolation:

```
predicted_position = position + VELOCITY_WEIGHT * (position - previous_position)
```
(falls back to `position` itself if there's no known predecessor yet; `VELOCITY_WEIGHT = 0.5`).

For each pair of consecutive frames `(t, t+1)`, build a cost matrix over all `(source, target)`
pairs within a distance gate:

```
cost(source, target) = motion_distance + 0.05 * raw_distance − LEARNED_BONUS * learned_edge_prob
```

where `motion_distance = ‖target_pos − predicted_position(source)‖`, `raw_distance =
‖target_pos − source_pos‖`, and `learned_edge_prob` is the node-transformer's score for that edge
(0 if the pair wasn't scored by the model), sigmoided if it looks like a raw logit outside `[0,1]`.
`LEARNED_BONUS = 1.0` in the selected calibration.

The assignment is solved **twice per frame pair**, greedily narrowing the pool each time
(`scipy.optimize.linear_sum_assignment`, i.e. the Hungarian algorithm, applied to the cost matrix
with unreachable pairs set to a very large "big" cost so they're never chosen):

1. **Tight pass**: gate = 6.0 µm.
2. **Relaxed pass**: gate = 10.0 µm, only over nodes still unmatched after the tight pass.

Each accepted match becomes a new edge carrying `edge_prob`, `distance_um`,
`motion_distance_um`, and metadata flags (`motion_relinked=1`, `motion_pass`). This produces a
graph where **each node has at most one predecessor and at most one successor per single-step
assignment** — divisions are *not* created here; they're added back deliberately and conservatively
in §4.8.

Safety valve: if any single frame has more than `MOTION_RELINK_MAX_FRAME_NODES` (2600) nodes, the
whole relink pass is skipped for that video and the raw filtered ILP edges are used instead
(`O(n²)` cost-matrix construction would be too expensive).

### 4.6 Single-parent repair

Even after motion-relink (which is already close to 1:1), a defensive repair pass keeps only the
single highest-scoring incoming edge per target node — score = `(edge_prob, -distance_um)`
lexicographic (prefer higher probability, then shorter distance). (The symmetric single-child
repair exists but is **off**, since it would prevent the deliberate division step below.)

### 4.7 Single-frame gap closing (`close_single_frame_gaps`)

Goal: repair tracks broken by exactly one missed detection frame.

1. Identify all track **ends** (nodes with no outgoing edge) and **starts** (nodes with no
   incoming edge), bucketed by frame.
2. `effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)` — **note the config requests `2` but the code
   clamps to `1`**, so in practice only single-frame gaps are bridged here (the notebook's own
   markdown explains this: two-frame gaps are handled by the separate, stricter `gap2` pass so a
   loose config override can't silently create non-consecutive edges).
3. For each frame `t`, match ends at `t` against starts at `t + 2` (gap=1) using
   `linear_sum_assignment` on pairwise physical distance, gated at `threshold_um = GAP_CLOSE_UM *
   (gap+1) = 11.6 µm`.
4. For each accepted `(end, start)` pair, compute the naive interpolated midpoint (average of the
   two positions), then resolve a middle node:
   - **Reuse an existing isolated node**: if there's a real, currently-untracked detection at the
     missing frame within `GAP_CLOSE_REUSE_UM` (3.2 µm) of the midpoint, reuse it as the bridge
     node (preferred — it's a real detection, not a fabrication).
   - **Otherwise synthesize a new node** (up to the per-video node budget:
     `min(GAP_CLOSE_MAX_ADDED_ABS, round(node_count * GAP_CLOSE_MAX_ADDED_FRAC))`), whose position
     is *refined* against the actual image data (`refine_synthetic_midpoint`, see next).
5. Insert two new edges: `end → middle` and `middle → start`.

**Pixel-level midpoint refinement** (`refine_synthetic_midpoint`): reads the raw frame from the
`.zarr` volume (with a manual blosc2-decompression fast path and a `zarr` library fallback),
crops a small window around the naive midpoint (`±1` voxel in Z, `±3` voxels in Y/X), subtracts
the local 20th-percentile intensity as background, and computes the **intensity-weighted centroid**
(a background-subtracted center-of-mass) of the remaining signal within that window. The refined
point replaces the naive midpoint only if it doesn't move more than `GAP_REFINE_MAX_SHIFT_UM`
(3.2 µm) away — otherwise the naive midpoint is kept and the rejection is logged to `run_stats`.

A dormant "DeepCenter" confirmation gate can veto a *reused* middle node here (see §6); it's
disabled in the scored run.

### 4.8 Strict 2-frame gap recovery (`recover_strict_gap2`) — present but disabled

Not active in the selected calibration (`BIOHUB_OUTPUT_GAP2_RECOVERY=0`), but fully implemented,
for exactly 2-missing-frame gaps (`start.t == end.t + 3`):

- Candidate pairs are gated on total distance (`GAP2_MAX_TOTAL_UM = 10.2 µm`) and average
  per-step distance (`GAP2_MAX_STEP_UM = 4.4 µm`).
- **Directional context check** (`GAP2_REQUIRE_CONTEXT=1`): the incoming motion vector into the
  gap's start and the outgoing motion vector out of its end must not point in a wildly different
  direction than the proposed gap-bridging vector (cosine similarity gate, `cos > -0.25`, with a
  soft penalty term added to the ranking score otherwise). This exists specifically to reject
  "coincidental" 3-frame-apart proximity that isn't part of the same real trajectory.
- Proposals are ranked by `distance + 2 * context_penalty` and greedily accepted under a global
  cap (`GAP2_MAX_LINKS_ABS=180`, or a fraction of total edges) and a per-frame cap.
- Two synthetic intermediate nodes are inserted per bridged gap (with the same pixel-refinement
  step as above), splitting the 3-frame gap into three consecutive 1-frame edges.

### 4.9 Conservative division insertion (`add_safe_divisions_postlink`)

This is the "safe-division" logic and the notebook explicitly frames it as intentionally
conservative: *"a high-quality single-child link is usually better than an unsupported second
child."* It only ever **adds** a second child edge — it never removes or alters the first.

For each frame `t`:

1. **Candidate parents** = nodes at `t` that currently have exactly one outgoing edge (single
   child).
2. **Candidate daughters** = nodes at `t+1` that currently have no incoming edge (unclaimed).
3. For each parent, its existing child link must already be tight:
   `distance(parent, existing_child) ≤ SAFE_DIV_EXISTING_CHILD_MAX_UM` (7.65 µm) — otherwise skip
   this parent entirely (don't add a division onto an already-shaky single-child link).
4. For each remaining `(parent, candidate)` pair:
   - `parent_dist = distance(parent, candidate) ≤ SAFE_DIV_MAX_UM` (4.66 µm)
   - `sister_dist = distance(existing_child, candidate) ≤ SAFE_DIV_SISTER_MAX_UM` (8.5 µm)
   - (optional, disabled) DeepCenter centerness confirmation on the candidate's location.
5. Surviving proposals are scored `parent_dist + 0.15 * sister_dist` (lower is better) and
   accepted **greedily, in score order**, subject to:
   - a per-frame cap (`round(num_parents_this_frame * SAFE_DIV_FRAME_FRAC_CAP)`, i.e. ~0.76% of
     that frame's eligible parents),
   - a global cap across the whole video (`round(num_edges * SAFE_DIV_GLOBAL_FRAC_CAP)`, ~0.375%),
   - each candidate daughter can only be claimed once.
6. Accepted proposals become new `parent → candidate` edges tagged `safe_division=1`.

### 4.10 Division geometry filter — present but disabled

`OUTPUT_DIVISION_GEOMETRY_FILTER=0`. If it were on, it would post-hoc inspect every node with more
than one outgoing edge, keep only the top-2 by `(edge_prob, -distance)`, and validate them against
`DIV_PARENT_MAX_UM` / `DIV_SISTER_MAX_UM`, collapsing to a single child if invalid
(`DIV_DROP_TO_SINGLE_IF_BAD=1`). It's redundant here because §4.9 only ever proposes divisions that
already satisfy tighter versions of the same geometric constraints — this filter exists as a
belt-and-suspenders safety net for configurations that skip §4.9's own gating.

### 4.11 Prune isolated nodes

Any node with no incident edges at all (source or target of nothing) after all the above is
dropped (`OUTPUT_PRUNE_ISOLATED=1`).

### 4.12 Short-track filtering (`filter_short_track_components`)

1. Build a **union-find (disjoint set)** structure over all nodes, unioning the endpoints of every
   edge — each resulting connected component is one full lineage tree (a single track, or a track
   plus its division sub-trees).
2. For each component, check whether *any* member node has out-degree ≥ 2 (a division exists
   anywhere in that lineage).
3. Keep a component if `size(component) ≥ OUTPUT_MIN_TRACK_LEN` (6) **or** it contains a division
   and `OUTPUT_KEEP_DIVISION_COMPONENTS=1` (true) — i.e. lineages that include a division are
   *never* removed by length, regardless of how short they are.
4. Everything else is dropped (nodes and their incident edges).

**Adaptive rescue** (`ADAPTIVE_SHORT_TRACK_RESCUE`, off in this run): if it were on and the
fraction of nodes removed by the above exceeded a trigger threshold (10%), it would re-admit some
short (but not too short) components back in, ranked by a quality score combining mean edge
probability, mean edge distance, and component size, under a node budget — a fallback to avoid
over-pruning on a video where the base detector/tracker was weak. It's not used in the selected
0.902 calibration.

### 4.13 Trajectory smoothing (`linefit_smooth_output_graph`)

A final, **topology-preserving** pass that only adjusts node *coordinates*, to reduce localization
jitter:

For each node, walk backward and forward along **unambiguous** single-predecessor/single-successor
chains up to `OUTPUT_LINEFIT_WINDOW` (2) steps in each direction, collecting up to 5 points
(`t-2..t+2`) with their relative time offsets. If at least 3 points are available:

1. Fit a **degree-1 polynomial (linear regression)** independently per axis (z, y, x) against the
   relative time offset.
2. Evaluate the fit at offset 0 to get a "fitted position" for the current node.
3. Blend: `new_position = (1 − w) * original + w * fitted`, with `w = OUTPUT_LINEFIT_WEIGHT = 0.8`
   — i.e. the node is moved 80% of the way toward what a locally-linear trajectory would predict.

Nodes without enough unambiguous local context (division points, gap edges, track ends) are left
untouched.

### 4.14 Writing `submission.csv` and `run_stats.csv`

Per video (`.geff` file), after the full calibration pipeline above:

- One CSV row per surviving **node**: `row_type="node"`, `node_id, t, z, y, x` (coordinates
  rounded to nearest int, clamped ≥ 0), `source_id=-1, target_id=-1`.
- One CSV row per surviving **edge**: `row_type="edge"`, `source_id, target_id`, with
  `node_id=-1, t=-1, z=-1, y=-1, x=-1`.
- Full column order: `id, dataset, row_type, node_id, t, z, y, x, source_id, target_id`.
- Rows are streamed directly to disk (`csv.DictWriter`) rather than buffered in memory, since the
  hidden test set can be large.

Sanity checks before finishing: every expected test dataset appears exactly once, no dataset
extra/missing, running row counter matches `total_nodes + total_edges`, and the CSV header matches
the required schema exactly.

Alongside the submission, `run_stats.csv` records **every single counter** touched by the
calibration pipeline per video — e.g. `dropped_nonconsecutive_edges, dropped_long_edges,
gap_pairs_selected, gap_inserted_synthetic, gap_refine_rejected_shift, motion_relink_tight_edges,
safe_divisions_added, short_track_nodes_removed, linefit_smoothed_nodes, deepcenter_gap_rejected,
...` — dozens of fields in total. This is what let the authors *prove*, rather than assume, that
the DeepCenter gate (§6) never rejected anything.

---

## 5. Practical takeaways (from the notebook's closing notes)

- **Validate the final graph, not just detector metrics.** A seemingly stronger detection
  checkpoint can be dominated by small changes in how associations are built downstream.
- **Treat divisions conservatively.** A high-confidence single-child link beats an unsupported
  second child — hence the tight, multiplicatively-gated, capped `add_safe_divisions_postlink`
  logic in §4.9 rather than trusting the ILP's raw division calls.
- **Keep post-processing deterministic and fully instrumented.** Emitting `run_stats.csv` for
  every repair/rejection/addition is what allowed the authors to *prove* the DeepCenter branch was
  dead code (checked zero nodes) and safely delete it from the public release, rather than leaving
  in an untested "maybe it helps" component.

---

## 6. The dormant DeepCenter veto (for completeness)

An earlier research variant of this pipeline loaded a **second, auxiliary 3D U-Net** ("DeepCenter")
whose only job was to predict a low-resolution "centerness" heatmap over each frame, and used it as
an *add-only* confirmation gate: before accepting a reused gap-middle node or a new division
daughter, it would check that location's predicted centerness score against a threshold
(`DEEPCENTER_GAP_THRESHOLD=0.20`, `DEEPCENTER_SAFE_DIV_THRESHOLD=0.12` in the config defaults) and
reject the addition if the score was too low.

Architecture (present in the code, `_DCDeepCenterUNet3D`): a small 3-level encoder-decoder 3D
U-Net — `Conv3d → GroupNorm → SiLU`, doubled per block, channels `24 → 48 → 96 → 192` at the
bottleneck, `MaxPool3d(2)` downsampling / `ConvTranspose3d(2,2)` upsampling with skip connections,
single-channel sigmoid output head. Input frames are downsampled by `pool_factor` (default 4) in
Y/X before being normalized (percentile-based dynamic range clipping) and passed through the net.

**Why it's disabled:** the `run_stats.csv` instrumentation showed this gate checked **zero** nodes
and edges in the actually-scored 0.902 artifact — every candidate bypassed it "by construction"
(the gating conditions in §4.7/§4.9 that would trigger a DeepCenter check never fired in practice
for this calibration). The public release therefore sets `BIOHUB_USE_DEEPCENTER_VETO=0`, which
short-circuits `load_deepcenter_veto_detector()` to return `None` immediately, and every downstream
`deepcenter_accept_repair_point(...)` call becomes a no-op `return True`. **You do not need this
component to recreate the 0.902 result** — it's included here only because its code is present in
the notebook and worth understanding if you want to extend the calibration further.

---

## 7. Recreating this from scratch — checklist

1. **Get or train a 3D nucleus detector** (referred to as "TemporalUNet3D") that outputs per-frame
   3D centroid candidates with a confidence score, at `DET_THRESHOLD=0.97`.
2. **Get or train a node-transformer edge scorer** that, given two candidate nodes in consecutive
   frames, outputs a learned probability that they're the same cell (`edge_prob`).
3. **Wire up an ILP tracker** (the notebook uses `tracksdata` + `ilpy`/`pyscipopt`) with cost
   weights `edge=-1.0, appearance=0.1, disappearance=0.1, division=1.0` to produce an initial
   candidate lineage graph per video, exported as `.geff`. (`motile`/`traccuracy`-style ILP
   tracking formulations are the standard prior art here if building this from scratch.)
4. **Apply the 8-way D4 TTA** at inference time on the detector (§4.3) — this is a small, well
   specified change with no external dependency beyond `torch.flip` / `torch.rot90` / `.transpose`.
5. **Copy the entire stage-4 calibration layer as-is** (§4.4–§4.13) — it's self-contained, only
   needs `{node_id: (t,z,y,x)}` + `[(source_id, target_id, edge_prob)]` as input, plus optional
   read access to the raw `.zarr` volumes for the pixel-refinement step in gap-closing. Reuse the
   exact thresholds in §3.1/§3.2 as your starting calibration — they were tuned against the public
   leaderboard already.
6. **Skip the DeepCenter gate** (§6) entirely — it added complexity and a whole second trained
   model for zero measured effect in the scored configuration.
7. **Instrument everything** the way `run_stats.csv` does — per-video counters for every
   add/drop/reject decision — so you can validate that each calibration knob is actually doing
   something before you rely on it.
