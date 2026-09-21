# Node Transformer — cross-frame edge scoring for motion graphs

New module: `repo/models/node_transformer.py`. This is stage (2) of the pipeline
described in `README.md` — the **node-transformer edge scorer** — which the
README documents as external/vendored (`pilkwang/biohub-tracking-support-pack-50ep-v1`,
not reproducible from that notebook). This is a from-scratch implementation
meant to sit directly on top of the `TemporalUNet3D` detector already trained
in this repo (`repo/models/temporal_unet.py`, weights in `weights/`).

**Training-ready.** `repo/train_edge.py` trains it end-to-end on top of the
pretrained `TemporalUNet3D` backbone (`weights/centroid_unet_best.pt`),
mirroring `train.py`'s exact paradigm (self-spawning DDP, wandb logging,
`EdgeTrainConfig` dataclass, best/last checkpointing). See "Training this"
below for how to run it, including on Kaggle.

## Where it sits in the pipeline

```
 raw .zarr (T=3 window)
        │
        ▼
 TemporalUNet3D + detect_head  ──►  per-frame centroid heatmaps  (existing, trained)
        │
        ▼
 peak extraction (local-max, see centroid_recall in train.py)  ──►  candidate centroids per frame
        │
        ▼
 NodeTransformer (THIS MODULE)  ──►  edge_prob for every geometrically-plausible
        │                            (frame t, frame t') candidate pair
        ▼
 assemble_candidate_graph  ──►  {nodes, edges} motion graph  ──► ILP / motion-relink (future stage)
```

The detector answers "where are the cells in this frame". This module
answers "which candidate in frame t is which candidate in frame t+1" —
the edges of the eventual lineage graph. It doesn't do assignment itself
(no Hungarian/ILP inside it); it produces calibrated pairwise probabilities
and leaves hard assignment (one parent, ≤2 children, appearance/disappearance
costs) to whatever consumes the candidate graph, same separation of concerns
as the reference pipeline (§4.5 of `README.md` folds `edge_prob` into its own
motion-relink cost, it doesn't invert a learned assignment).

## Why it's coupled to the detector backbone

Rather than a separate image encoder, node embeddings start from features
sampled out of the detector's own feature maps:

```python
feats = centroid_unet.unet(imgs)          # (B, T, C, Z, Y, X) — already computed for detection
node_feats = sample_node_features(feats, coords)   # (B, T, N, C), trilinear grid_sample at each centroid
```

This is free (the detector already computes `feats` on the forward pass that
produced the centroids in the first place) and means the edge scorer inherits
whatever the detector already learned about local appearance — it only has to
learn *matching*, not re-learn nucleus appearance from scratch.

## Architecture

1. **Node embedding.** For each candidate centroid: `sample_node_features`
   (backbone feature, trilinear-sampled) + a Fourier positional encoding of
   its physical `(z, y, x)` position in microns (`_fourier_features`, fixed
   sinusoids at 8 log-spaced periods per axis, so it's well-defined outside
   the training coordinate range) + a learned per-slot time embedding
   (`nn.Embedding(window_size, embed_dim)` — window position 0/1/2, not
   absolute video time).
2. **Cross-frame self-attention.** All `T*N` node embeddings (real + padding)
   go through a standard `nn.TransformerEncoder` (pre-LN, GELU, `n_layers=3`
   by default) with a `src_key_padding_mask` derived from the candidate mask.
   Every node can attend to every other node in the window regardless of
   frame — a node at t sees candidates at both t+1 and t+2 directly, so
   long-range motion cues aren't stuck being relayed frame-by-frame.
3. **Edge scoring.** For each scored frame pair `(t_a, t_b)` — every
   consecutive pair, plus `t -> t+2` pairs when `skip_frame_edges=True`
   (default; these are what let a future gap-closing step recover a
   single missed detection, mirroring `README.md` §4.7) — score every
   `(i, j)` candidate pair as:

   ```
   logit(i, j) = scale * <edge_query(embed[t_a, i]), edge_key(embed[t_b, j])>
               + geo_bias_mlp([dz, dy, dx, |d|, t_b - t_a])
   ```

   the first term is a standard scaled-dot-product attention score; the
   second is a learned function of the raw physical offset — a learned
   analogue of the reference pipeline's hand-tuned
   `motion_distance + 0.05*raw_distance - bonus*edge_prob` cost (§4.5), except
   here the model learns the tradeoff instead of it being a tuned constant.
4. **Distance gating.** Pairs farther apart than `max_link_distance_um`
   (default 14.0, matching `README.md`'s `OUTPUT_EDGE_MAX_UM`) are masked to
   `-inf` before scoring — never even given a chance to be a false positive.
   This keeps the pairwise score matrix cheap (candidates per frame are far
   fewer than voxels) and rules out physically-impossible matches by
   construction rather than hoping the network learns to reject them.
5. **Zero-init.** `edge_key` and the last layer of `geo_bias_mlp` are
   zero-initialized, so every valid candidate pair starts at logit 0 (prob
   0.5) before training — the same "start uninformative, let training pull
   real matches away from distractors" rationale as the deformable temporal
   attention's zero-init in `temporal_unet.py`.

## Input/output contract

Deliberately matches the tensors `train.py`'s `CentroidWindowDataset` /
`collate_windows` already produce, so it can consume GT centroids directly
during training or peak-extracted centroids at inference without reshaping:

```python
out = node_transformer(feats, coords, mask, voxel_size)
# feats  : (B, T, C, Z, Y, X)  — CentroidUNet.unet(imgs) output
# coords : (B, T, N, 3)        — candidate (z, y, x) in the SAME downsampled voxel space as feats
# mask   : (B, T, N)  bool     — True where a candidate is real, not padding
# voxel_size : (vz, vy, vx) microns per voxel (VideoMeta.voxel_size)

out["embeddings"]   # (B, T, N, D)
out["edge_logits"]  # {(t_a, t_b): (B, N, N)}, -inf where invalid
out["valid_masks"]  # {(t_a, t_b): (B, N, N)} bool, matches edge_logits' finite entries
```

`sigmoid(out["edge_logits"][(t_a, t_b)][b, i, j])` is the `edge_prob` for
"candidate i in frame t_a is the same cell as candidate j in frame t_b".

## Loss + metrics

`edge_bce_loss` and `edge_precision_recall` are direct analogues of
`detection_loss`/`centroid_recall` already in `train.py`:

- `edge_bce_loss(logits, pos_mask, valid_mask, neg_weight=0.1)` — BCE with the
  same per-sample reweighting scheme as `detection_loss`: all positive pairs
  in a sample sum to weight 1, all valid negatives share `neg_weight` — true
  matches are a tiny fraction of geometrically-plausible candidate pairs,
  same imbalance `detection_loss` corrects for between GT voxels and
  background.
- `edge_focal_loss(logits, pos_mask, valid_mask, gamma=2.0)` — what
  `train_edge.py` actually trains with. Column-softmax (over the *source*
  axis) + focal BCE, matching pilkwang's vendored `train_unet_transformer.py`
  loss (`compute_loss`). Normalizing over sources per target column encodes
  "at most one true parent per target" (no merges) while leaving the source
  axis free to score high against multiple targets (divisions are fine).
  This is a materially different inductive bias than `edge_bce_loss`'s
  independent per-pair sigmoids — see "Why the loss changed" below.
- `edge_precision_recall(logits, pos_mask, valid_mask, threshold=0.0)` — raw
  TP/FP/FN counts at a given logit threshold (0.0 = prob 0.5), for computing
  precision/recall/F1 during validation. Works the same regardless of which
  loss trained the model.

All three take `pos_mask: (B, N, M)` bool — ground-truth same-cell pairs —
which `train_edge.py` builds directly from each window's `.geff` edges (see
below); `edge_bce_loss` is kept as a simpler independent-sigmoid alternative
but isn't what `train_edge.py` uses by default.

## Building the motion graph

`assemble_candidate_graph(coords, mask, edge_logits, t_start, threshold)`
converts one (unbatched) window's scored candidates into the plain-dict
candidate-graph shape `README.md` §4.2 describes the reference pipeline
consuming (`nodes_by_id` / `raw_edges`):

```python
{"nodes": {(t, local_idx): (abs_t, z, y, x), ...},
 "edges": [((t_a, i), (t_b, j), edge_prob), ...]}
```

This is the "motion graph from predicted centers across timescales" the
module is for. It's intentionally *not* wired to a specific tracker yet:
hand it to an ILP formulation (as the reference pipeline does — `tracksdata`
+ `ilpy`, cost weights in `README.md` §3.1/§7), or do a greedy/Hungarian
per-frame-pair assignment directly on `edge_prob`, or feed it into a
from-scratch reimplementation of the reference pipeline's stage 4 motion-relink
+ gap-closing calibration layer (`README.md` §4.4–§4.13), which already
expects exactly this `{node_id: (t,z,y,x)}` + `[(source, target, edge_prob)]`
shape as input. Overlapping windows (`window_stride < WINDOW_SIZE`) will
propose the same edge more than once when stitched together — dedupe by
`(source, target)` keeping the max `edge_prob`.

## Training this — `repo/train_edge.py`

Mirrors `train.py`'s exact paradigm: self-spawning DDP (no `torchrun`
needed), wandb logging from rank 0, a no-CLI `EdgeTrainConfig` dataclass you
build and pass to `run()`, best/last checkpointing. All new dataset/model
code lives in `train_edge.py`; it imports and reuses `train.py`'s
`CentroidUNet`, `detection_loss`, `centroid_recall`, `DEFAULT_AUGMENTATIONS`,
and the DDP helpers directly rather than duplicating them.

**Data pipeline (`load_edge_video_windows`, `EdgeWindowDataset`).** Same
windowing as `train.py`'s `load_video_windows`, but also indexes each
video's `.geff` edges by source frame and, for every window and every
scoreable frame-pair offset `(i, j)` (all of `itertools.combinations(range(3), 2)`
= `{(0,1), (1,2), (0,2)}` for `WINDOW_SIZE=3` — exactly the pairs
`NodeTransformer` scores with `skip_frame_edges=True`), builds a
`(K, 2)` array of local `(source_idx, target_idx)` pairs from real graph
edges with `t[u] == t_start+i` and `t[v] == t_start+j`. Since `.geff` only
stores real edges (not a transitive closure), this naturally captures both
consecutive-frame lineage and genuine gap edges (annotator skipped a frame
for that cell) with the same lookup. `train.py`'s existing augmentations
(flip, rot90, brightness, contrast/gamma, noise) are reused unchanged — they
only ever transform coordinate *values* or flip the image, never reorder the
node axis, so index-based positive pairs stay valid after augmentation.

**Trains on GT centroids, not detector peaks** (teacher forcing) — a
deliberate simplification vs. pilkwang's vendored recipe, which trains on
the detector's own live peaks (run detection → NMS peak-extract → greedy
match to GT → propagate GT edges through the match) every step. Teacher
forcing is simpler and has no risk of degenerate batches (e.g. zero peaks
early in training); the cost is a train/inference distribution mismatch,
since inference still runs on peak-extracted detections. Worth revisiting
with pilkwang's live-detect-and-match loop as a follow-up once this baseline
is validated (see "Known limitations").

**Backbone fine-tunes jointly by default** (`freeze_backbone=False`),
matching pilkwang's end-to-end recipe rather than a frozen-features
approach: `EdgeModel` wraps the pretrained `CentroidUNet` + `NodeTransformer`
and backprops the edge loss straight into the backbone, plus an auxiliary
`detection_loss` (weight `det_loss_weight=0.1`) so the backbone doesn't
drift away from what made it good at detection while it adapts for
matching. Set `freeze_backbone=True` for a faster, lower-risk run that only
trains `NodeTransformer` (no detection loss computed in that mode — nothing
to train the head with).

**Why the loss changed**: `train_edge.py` trains with `edge_focal_loss`
(column-softmax + focal, pilkwang's formulation — see above), not
`edge_bce_loss`. One consequence worth knowing before reading validation
curves: for a target with exactly one valid candidate parent (no real
competition — common in this dataset's sparser videos, e.g.
`44b6_0113de3b.geff` has ~1 node/frame throughout), softmax over a
single-element axis is trivially 1.0 regardless of the logit, so that pair
contributes exactly zero loss/gradient. This is correct, not a bug — there's
no decision to learn when there's only one option — but it means the
reported edge loss on sparse windows can be exactly 0.0 even from an
untrained model. Busier videos (some have up to 11 cells/frame) are where
the loss actually has signal.

**Smoke-tested with real data** (not just random tensors): pretrained
`weights/centroid_unet_best.pt` loads into `CentroidUNet` with `strict=True`
(architectures match exactly), a full forward+backward step was verified to
update both `NodeTransformer` and backbone parameters (diffed before/after),
and `evaluate()` produces finite, non-degenerate precision/recall on both a
sparse and a busy real video from `data/train/`.

### Running it

```python
from train_edge import EdgeTrainConfig, run

run(EdgeTrainConfig(
    data_dir="data/train",                              # dir of matching <id>.zarr / <id>.geff
    unet_weights="weights/centroid_unet_best.pt",        # required: stage-1 detector checkpoint
    output_dir="weights",                                # writes edge_model_{best,last}.pt here
    n_epochs=50,
    lr=1e-4,
    batch_size=4,                                        # per-GPU
    freeze_backbone=False,                                # True = faster/safer, skips detection loss
    wandb_project="biohub-cell-tracking",
    wandb_run_name="edge-v1",
))
```

`unet_out_channels` / `unet_layers` / `unet_n_heads` / `unet_n_points` in
`EdgeTrainConfig` must match whatever `TrainConfig` was used to produce
`unet_weights` (defaults on both sides already agree: `32`, `[32,64,128]`,
`4`, `4`) — `centroid_unet.load_state_dict(state, strict=True)` will raise
immediately if they don't.

**Logged to wandb** (project `biohub-cell-tracking` by default, same as
`train.py`): per-step `train/loss_step`, `train/edge_loss_step`,
`train/det_loss_step`, `train/grad_norm`, `train/lr`; per-epoch
`train/loss_epoch`, `train/edge_loss_epoch`, `train/det_loss_epoch`,
`val/loss`, `val/edge_loss`, `val/det_loss`, `val/edge_precision`,
`val/edge_recall`, `val/edge_f1`, `val/best_edge_f1`, `val/det_recall` (only
when `freeze_backbone=False`), plus `data/n_train_windows`,
`data/n_val_windows`, `model/n_trainable_params`. Checkpoint selection is by
best `val/edge_f1`, same pattern as `train.py` selecting on `val/recall`.

### Running on Kaggle

Same shape as how `train.py` expects to be run (see its docstring), extended
with the extra pretrained-backbone dependency:

1. **Attach two datasets** to the notebook: the competition data (for
   `<id>.zarr`/`<id>.geff`) and a dataset containing this repo's
   `weights/centroid_unet_best.pt` (upload it as a private Kaggle dataset
   once, or re-run `train.py`'s stage first in the same session).
2. **Set the wandb API key** as a Kaggle secret (Add-ons → Secrets →
   `WANDB_API_KEY`), then in the notebook:
   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
   ```
   (skip this and pass `wandb_mode="offline"` or `"disabled"` if you don't
   want to log during the run).
3. **Install deps** (same as this repo's inference environment) and put the
   repo on `sys.path`:
   ```python
   !pip install -q geff zarr wandb
   import sys; sys.path.append("/kaggle/working/BioHub-Kaggle/repo")
   ```
4. **Run**:
   ```python
   from train_edge import EdgeTrainConfig, run

   run(EdgeTrainConfig(
       data_dir="/kaggle/input/biohub-cell-tracking-during-development/train",
       unet_weights="/kaggle/input/<your-backbone-weights-dataset>/centroid_unet_best.pt",
       output_dir="/kaggle/working/weights",
       n_epochs=50,
       n_gpus=2,          # match the accelerator (e.g. "GPU T4 x2"); None = auto-detect all visible
       wandb_mode="online",
   ))
   ```
5. **After training**, `/kaggle/working/weights/edge_model_{best,last}.pt`
   holds the full `EdgeModel` state dict (backbone + `NodeTransformer`
   together) — load it back with the same `CentroidUNet`/`NodeTransformer`
   construction shown in `train_edge.train()` before feeding real detector
   output through `assemble_candidate_graph` for the ILP stage.

## Known limitations

- Trains on GT centroids (teacher forcing), not the detector's own peaks —
  see "Trains on GT centroids, not detector peaks" above. The ILP stage that
  consumes this module's output at inference time will still see real
  peak-extraction noise the model never saw during training.
- Independent per-pair sigmoids in `edge_bce_loss` (still available, not the
  default) vs. `edge_focal_loss`'s column-softmax — the latter structurally
  forbids merges (one parent per target) but doesn't forbid a node scoring
  high against multiple partners across different target columns, which is
  what a division needs. Neither loss enforces global consistency (e.g. two
  different targets both claiming the same single-candidate parent isn't
  prevented); that's still left to the downstream ILP/tracker, matching the
  reference pipeline's own separation of concerns.
- `t_b - t_a` is a raw frame-count in `geo_bias_mlp`, not physical time —
  fine while every window is uniformly-spaced `WINDOW_SIZE=3` frames, would
  need revisiting for irregular frame spacing.
- Verified end-to-end on real data with a tiny number of steps (forward
  shapes, strict weight loading, gradient flow into both submodels, finite
  loss/metrics) — not a full training run, so hyperparameters
  (`embed_dim=128`, `n_layers=3`, `max_link_distance_um=14.0`,
  `det_loss_weight=0.1`, `focal_gamma=2.0`) are reasonable priors from the
  reference pipeline's tuned constants, not empirically tuned for this
  implementation.
