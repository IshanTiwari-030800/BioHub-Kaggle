# Node Transformer — cross-frame edge scoring for motion graphs

New module: `repo/models/node_transformer.py`. This is stage (2) of the pipeline
described in `README.md` — the **node-transformer edge scorer** — which the
README documents as external/vendored (`pilkwang/biohub-tracking-support-pack-50ep-v1`,
not reproducible from that notebook). This is a from-scratch implementation
meant to sit directly on top of the `TemporalUNet3D` detector already trained
in this repo (`repo/models/temporal_unet.py`, weights in `weights/`).

Not trained yet — this is the architecture + loss/metric utilities + a
graph-assembly helper. See "Training this" below for what's still needed.

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
- `edge_precision_recall(logits, pos_mask, valid_mask, threshold=0.0)` — raw
  TP/FP/FN counts at a given logit threshold (0.0 = prob 0.5), for computing
  precision/recall/F1 during validation.

Both take `pos_mask: (B, N, M)` bool — ground-truth same-cell pairs — which
the caller has to build from the `.geff` graph's edges (see below).

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

## Training this (not yet implemented — sketch for next step)

No `train_edge.py` yet. The pieces above are enough to write one closely
mirroring `train.py`'s structure:

1. **Reuse `CentroidWindowDataset`/`load_video_windows`** for windowed
   images + GT centroids — already gives `(imgs, coords, mask)` per window.
2. **Positive pairs**, per window, come straight from the `.geff` graph's
   real edges (not from any ID-encoding convention — read them directly):

   ```python
   graph, _ = geff.read(str(geff_path), node_props=["t"])
   # index GT centroids in coords[t_a] / coords[t_b] the same order load_video_windows built them in,
   # then for each geff edge (u, v) with t[u] == t_start + t_a and t[v] == t_start + t_b:
   #     pos_mask[b, index_of(u), index_of(v)] = True
   ```

   `load_video_windows` builds `window.coords[i]` straight from
   `graph.nodes(data=True)` per frame — keep that same per-frame node order
   (and the underlying geff node ids) around when building windows so edges
   can be looked up by index instead of re-deriving them from node ids.
3. **Train on GT centroids, not detector peaks** (teacher forcing), same
   reasoning `train.py` uses GT voxels for `detection_loss` rather than the
   model's own noisy peaks — matches during training should reflect true
   cell identity, not detector noise. Optionally fine-tune end-to-end with
   detector peaks later once both stages work independently.
4. **Freeze vs. fine-tune the backbone**: cheapest path is loading
   `weights/centroid_unet_best.pt`, freezing `CentroidUNet.unet`, and only
   training `NodeTransformer` on top (fast, no risk of degrading detection
   recall). Unfreezing later is a valid follow-up if edge accuracy plateaus
   on backbone features that weren't trained to be matching-discriminative.
5. **Loss**: `sum(edge_bce_loss(...) for each scored pair) / n_pairs`, same
   pattern as `train.py`'s per-frame loss averaging.
6. **Metric**: precision/recall/F1 from `edge_precision_recall`, aggregated
   the same way `train.py` reduces `centroid_recall`'s counts across
   ranks/batches before dividing.

## Known limitations

- Independent per-pair sigmoids, not a joint assignment — a node can score
  high against multiple partners (expected/useful for divisions, but note
  the module itself doesn't enforce "at most one parent"; that's left to
  the downstream tracker, matching the reference pipeline's own
  single-parent-repair being a *separate* pass, §4.6).
- `t_b - t_a` is a raw frame-count in `geo_bias_mlp`, not physical time —
  fine while every window is uniformly-spaced `WINDOW_SIZE=3` frames, would
  need revisiting for irregular frame spacing.
- Only smoke-tested (forward shapes, zero-init, gradient flow, loss/metric
  sanity, graph assembly) with random tensors — no training run yet, so
  hyperparameters (`embed_dim=128`, `n_layers=3`, `max_link_distance_um=14.0`)
  are reasonable defaults/priors from the reference pipeline's tuned
  constants, not empirically tuned for this implementation.
