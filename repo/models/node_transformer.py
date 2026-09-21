"""Cross-frame node attention for scoring cell-lineage edges over a T=3 window.

This is the "node-transformer edge scorer" described as stage (2) of the
pipeline in ``README.md``: given per-frame candidate cell centroids (produced
by ``TemporalUNet3D`` + peak extraction, see ``train.py``/``CentroidUNet``)
it predicts, for every geometrically-plausible pair of candidates in
consecutive (or near-consecutive) frames, the probability that they are the
same cell. Those probabilities (``edge_prob``) are exactly the quantity the
downstream ILP tracker in the reference pipeline consumes as its edge cost,
and are what ``assemble_candidate_graph`` below turns into an actual motion
graph.

Design, at a glance:

  1. ``sample_node_features`` pulls a feature vector for every candidate
     centroid out of the detector backbone's feature maps (the same
     ``TemporalUNet3D`` features the detection head reads, sampled at the
     continuous centroid location via trilinear ``grid_sample``) — no
     separate image encoder is trained.
  2. Each node embedding is that feature vector plus a Fourier positional
     encoding of its physical (z, y, x) position and a learned per-slot
     time embedding, then refined by a standard multi-head self-attention
     encoder over *every* node in the window jointly (so a node at t can
     directly attend to candidates at t+1 *and* t+2, not just its own
     frame).
  3. A scaled dot-product scoring head (query from frame t, key from frame
     t') plus a small MLP bias over the raw relative offset (a learned
     motion prior, in the spirit of the reference pipeline's hand-tuned
     motion-relink cost) produces one edge logit per candidate pair.
     Candidates further apart than ``max_link_distance_um`` are masked out
     before any loss/inference, the same physical-distance gating the
     reference pipeline's motion-relink and gap-closing stages use.

Built for ``WINDOW_SIZE = 3`` (imported from ``datasets.py``) but not
hard-coded to it: the time-slot embedding table is sized by ``window_size``,
so a 2- or 4-frame window works with the same code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import WINDOW_SIZE


def _fourier_features(x: torch.Tensor, n_bands: int = 8, max_period_um: float = 200.0) -> torch.Tensor:
    """Sinusoidal encoding of a scalar physical coordinate (in microns).

    ``n_bands`` log-spaced periods from ``max_period_um`` down to
    ``max_period_um / 2**(n_bands - 1)``, each contributing a (sin, cos)
    pair. Fixed (non-learned), so the encoding is well-defined for
    positions outside the range seen during training.
    """
    device, dtype = x.device, x.dtype
    periods = max_period_um * (2.0 ** -torch.arange(n_bands, device=device, dtype=dtype))
    freqs = (2.0 * math.pi) / periods  # (n_bands,)
    args = x.unsqueeze(-1) * freqs  # (..., n_bands)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (..., 2 * n_bands)


def sample_node_features(feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample per-node feature vectors from dense backbone features.

    Parameters
    ----------
    feats : (B, T, C, Z, Y, X)
        Backbone feature maps, e.g. ``CentroidUNet.unet(imgs)`` output
        (before the 1x1x1 detection head).
    coords : (B, T, N, 3)
        Continuous voxel coordinates ``(z, y, x)`` per candidate node, same
        convention as ``train.py``'s GT/predicted centroid tensors. Padding
        rows (see ``mask``) may hold any value — they are sampled too but
        the caller is expected to mask them out downstream.

    Returns
    -------
    (B, T, N, C) sampled feature vectors.
    """
    B, T, C, Z, Y, X = feats.shape
    N = coords.shape[2]

    z, y, x = coords[..., 0], coords[..., 1], coords[..., 2]
    # align_corners=False convention (matches temporal_unet._normalized_coords),
    # extended to continuous (non-integer) coordinates.
    gz = -1.0 + (2.0 * z + 1.0) / Z
    gy = -1.0 + (2.0 * y + 1.0) / Y
    gx = -1.0 + (2.0 * x + 1.0) / X
    grid = torch.stack([gx, gy, gz], dim=-1).reshape(B * T, 1, 1, N, 3)

    vol = feats.reshape(B * T, C, Z, Y, X)
    sampled = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # sampled: (B*T, C, 1, 1, N)
    return sampled.view(B, T, C, N).permute(0, 1, 3, 2)  # (B, T, N, C)


class NodeTransformer(nn.Module):
    """Cross-frame node self-attention + pairwise edge scoring over a T-window.

    Parameters
    ----------
    feat_channels : int
        Channel count of the backbone feature maps fed to ``forward``
        (``unet_out_channels`` in ``train.py``'s ``TrainConfig``).
    embed_dim : int
        Node embedding width used throughout the transformer encoder.
    n_heads, n_layers, ffn_dim, dropout :
        Standard ``nn.TransformerEncoder`` hyperparameters.
    window_size : int
        Number of frames per window (``T``); sizes the time-slot embedding
        table. Defaults to the project-wide ``WINDOW_SIZE`` (3).
    n_fourier_bands : int
        Bands per axis in the positional encoding (see ``_fourier_features``);
        the position MLP input is ``3 * 2 * n_fourier_bands``.
    max_link_distance_um : float
        Default physical-distance gate for candidate pairs (overridable
        per call to ``forward``). Candidates farther apart are never
        scored — mirrors the reference pipeline's motion-relink /
        gap-closing distance gates (``README.md`` §3.1-3.2).
    skip_frame_edges : bool
        If True (default), also score ``t -> t+2`` pairs in addition to
        every consecutive ``t -> t+1`` pair. These help recover single-frame
        missed detections (the reference pipeline's gap-closing stage) and
        cost little extra compute since they reuse the same node embeddings.
    """

    def __init__(
        self,
        feat_channels: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        window_size: int = WINDOW_SIZE,
        n_fourier_bands: int = 8,
        max_link_distance_um: float = 14.0,
        skip_frame_edges: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.max_link_distance_um = max_link_distance_um
        self.skip_frame_edges = skip_frame_edges
        self.n_fourier_bands = n_fourier_bands

        self.input_proj = nn.Linear(feat_channels, embed_dim)
        self.pos_proj = nn.Linear(3 * 2 * n_fourier_bands, embed_dim)
        self.time_embed = nn.Embedding(window_size, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.edge_query = nn.Linear(embed_dim, embed_dim)
        self.edge_key = nn.Linear(embed_dim, embed_dim)
        self.geo_bias_mlp = nn.Sequential(
            nn.Linear(5, embed_dim // 2),  # (dz, dy, dx, |d|, delta_t) in microns/frames
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self._scale = embed_dim ** -0.5

        # Zero-init both the dot-product scorer and the geometric-bias MLP's
        # output layer, so every valid candidate pair starts at logit 0
        # (prob 0.5, uninformative) rather than an arbitrary random score —
        # same "start uniform, let training pull apart real matches from
        # distractors" rationale as the deformable attention's zero-init in
        # temporal_unet.py.
        nn.init.zeros_(self.edge_key.weight)
        nn.init.zeros_(self.edge_key.bias)
        nn.init.zeros_(self.geo_bias_mlp[-1].weight)
        nn.init.zeros_(self.geo_bias_mlp[-1].bias)

    def _encode_nodes(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        voxel_size: tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (embeddings (B,T,N,D), positions_um (B,T,N,3))."""
        B, T, N = coords.shape[:3]
        vs = torch.as_tensor(voxel_size, device=coords.device, dtype=coords.dtype)
        pos_um = coords * vs  # (B, T, N, 3), (z, y, x) in microns

        node_feats = sample_node_features(feats, coords)  # (B, T, N, C)
        pos_enc = self.pos_proj(_fourier_features(pos_um, self.n_fourier_bands).reshape(B, T, N, -1))
        time_idx = torch.arange(T, device=coords.device)
        time_enc = self.time_embed(time_idx).view(1, T, 1, -1)

        x = self.input_norm(self.input_proj(node_feats) + pos_enc + time_enc)
        x = x.reshape(B, T * N, -1)
        key_padding_mask = ~mask.reshape(B, T * N)  # True = ignore, per nn.Transformer convention

        # A fully-padded sample (window with zero candidates in every frame)
        # would give every row of key_padding_mask True, which some PyTorch
        # versions turn into NaNs; guard by never fully masking a row.
        all_masked = key_padding_mask.all(dim=1)
        key_padding_mask = key_padding_mask & ~all_masked.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x.reshape(B, T, N, -1), pos_um

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        voxel_size: tuple[float, float, float],
        max_link_distance_um: float | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        feats : (B, T, C, Z, Y, X) backbone feature maps.
        coords : (B, T, N, 3) candidate centroid voxel coords, (z, y, x).
        mask : (B, T, N) bool, True where a candidate is real (not padding).
        voxel_size : physical (Z, Y, X) size of one voxel, in microns.
        max_link_distance_um : overrides the constructor default.

        Returns
        -------
        dict with:
          ``embeddings``  : (B, T, N, D) refined per-node embeddings.
          ``edge_logits`` : {(t_a, t_b): (B, N, N)} raw scores for every
            scored frame pair (all consecutive pairs, plus t -> t+2 pairs
            if ``skip_frame_edges``). Entry ``[b, i, j]`` is the logit for
            "node i in frame t_a is the same cell as node j in frame t_b";
            ``sigmoid(logit)`` is the ``edge_prob`` the reference pipeline's
            ILP/motion-relink stage expects. Invalid pairs (padding on
            either side, or beyond the distance gate) are ``-inf``.
          ``valid_masks``  : {(t_a, t_b): (B, N, N)} bool, matching
            ``edge_logits`` — pairs where the logit is finite.
        """
        gate = max_link_distance_um if max_link_distance_um is not None else self.max_link_distance_um
        B, T, N = coords.shape[:3]
        embed, pos_um = self._encode_nodes(feats, coords, mask, voxel_size)

        pairs = [(t, t + 1) for t in range(T - 1)]
        if self.skip_frame_edges:
            pairs += [(t, t + 2) for t in range(T - 2)]

        edge_logits: dict[tuple[int, int], torch.Tensor] = {}
        valid_masks: dict[tuple[int, int], torch.Tensor] = {}

        for t_a, t_b in pairs:
            q = self.edge_query(embed[:, t_a])  # (B, N, D)
            k = self.edge_key(embed[:, t_b])    # (B, N, D)
            dot = torch.einsum("bnd,bmd->bnm", q, k) * self._scale

            rel = pos_um[:, t_b].unsqueeze(1) - pos_um[:, t_a].unsqueeze(2)  # (B, N, N, 3), j - i
            dist = rel.norm(dim=-1)  # (B, N, N)
            dt = torch.full_like(dist[..., None], float(t_b - t_a))
            geo_bias = self.geo_bias_mlp(torch.cat([rel, dist.unsqueeze(-1), dt], dim=-1)).squeeze(-1)

            logits = dot + geo_bias
            valid = mask[:, t_a].unsqueeze(2) & mask[:, t_b].unsqueeze(1) & (dist <= gate)
            logits = logits.masked_fill(~valid, float("-inf"))

            edge_logits[(t_a, t_b)] = logits
            valid_masks[(t_a, t_b)] = valid

        return {"embeddings": embed, "edge_logits": edge_logits, "valid_masks": valid_masks}


# =============================================================================
# Loss + metrics (mirrors detection_loss / centroid_recall in train.py)
# =============================================================================

def edge_bce_loss(
    logits: torch.Tensor,      # (B, N, M)
    pos_mask: torch.Tensor,    # (B, N, M) bool, GT same-cell pairs
    valid_mask: torch.Tensor,  # (B, N, M) bool, candidate is scoreable (finite logit)
    neg_weight: float = 0.1,
) -> torch.Tensor:
    """BCE over one scored frame pair, GT-imbalance-reweighted like ``detection_loss``:

    positive pairs sum to weight 1 per sample, valid negative pairs
    (candidates within the distance gate that aren't a true match) share
    ``neg_weight`` per sample — true matches are rare relative to the
    number of geometrically-plausible candidate pairs, same imbalance
    ``detection_loss`` corrects for between GT voxels and background.
    """
    B = logits.shape[0]
    pos_mask = pos_mask & valid_mask
    target = pos_mask.float()

    n_pos = pos_mask.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (valid_mask & ~pos_mask).reshape(B, -1).sum(dim=1).clamp(min=1)
    shape = (B, 1, 1)
    weight = torch.where(pos_mask, (1.0 / n_pos).reshape(shape), (neg_weight / n_neg).reshape(shape))
    weight = weight * valid_mask.float()

    logits_safe = torch.where(valid_mask, logits, torch.zeros_like(logits))
    return F.binary_cross_entropy_with_logits(logits_safe, target, weight=weight, reduction="sum") / B


def edge_focal_loss(
    logits: torch.Tensor,      # (B, N, M), N=source frame candidates, M=target frame candidates
    pos_mask: torch.Tensor,    # (B, N, M) bool, GT same-cell pairs
    valid_mask: torch.Tensor,  # (B, N, M) bool, candidate is scoreable (finite logit)
    gamma: float = 2.0,
) -> torch.Tensor:
    """Column-softmax + focal BCE, matching pilkwang's ``compute_loss`` (the
    vendored ``train_unet_transformer.py`` this project extends to T=3).

    Softmax is taken over the *source* axis (dim=1) so every target
    candidate's incoming probability mass sums to 1 across its competing
    parents — encodes "at most one true parent per target" (no merges)
    while leaving the source axis unconstrained, so one source can still
    score high against two targets (a division). This is a materially
    different inductive bias than ``edge_bce_loss``'s independent
    per-pair sigmoids, which enforce neither constraint.

    Only GT-annotated rows/columns are supervised (``active_mask``): most
    of a distance-gated candidate matrix has no annotated node on one side
    at all, and those cells carry no information about matching, same
    "unannotated cells ignored" reasoning as the reference implementation.
    """
    logits = logits.masked_fill(~valid_mask, float("-inf"))
    probs = torch.softmax(logits, dim=1)
    probs = torch.nan_to_num(probs, nan=0.0)  # target columns with zero valid sources -> softmax over all -inf

    target = (pos_mask & valid_mask).float()
    active_rows = (pos_mask & valid_mask).any(dim=2)  # (B, N)
    active_cols = (pos_mask & valid_mask).any(dim=1)  # (B, M)
    active_mask = (active_rows.unsqueeze(2) | active_cols.unsqueeze(1)) & valid_mask

    bce = F.binary_cross_entropy(probs.clamp(1e-6, 1.0 - 1e-6), target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    focal = (1 - p_t) ** gamma

    B = logits.shape[0]
    denom = active_mask.reshape(B, -1).sum(dim=1).clamp(min=1)
    per_sample = (focal * bce * active_mask.float()).reshape(B, -1).sum(dim=1) / denom
    return per_sample.mean()


@torch.no_grad()
def edge_precision_recall(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.0,  # logit space; 0.0 == prob 0.5
) -> tuple[int, int, int]:
    """Returns (true_positives, false_positives, false_negatives) counts."""
    pos_mask = pos_mask & valid_mask
    pred = (logits > threshold) & valid_mask
    tp = int((pred & pos_mask).sum().item())
    fp = int((pred & ~pos_mask).sum().item())
    fn = int((pos_mask & ~pred).sum().item())
    return tp, fp, fn


# =============================================================================
# Turning scored candidates into an actual motion graph
# =============================================================================

@torch.no_grad()
def assemble_candidate_graph(
    coords: torch.Tensor,   # (T, N, 3) voxel coords for ONE sample, one window
    mask: torch.Tensor,     # (T, N) bool
    edge_logits: dict[tuple[int, int], torch.Tensor],  # {(t_a,t_b): (N, N)}, unbatched
    t_start: int = 0,
    threshold: float = 0.0,
) -> dict:
    """Convert one window's scored candidates into the plain-dict candidate
    graph described in ``README.md`` §4.2 (``nodes_by_id`` / ``raw_edges``):
    the same shape of object the reference pipeline's ILP tracker consumes,
    and what's written out as ``.geff``. Runs on a single (unbatched)
    sample; callers loop over the batch and merge/stitch windows themselves
    (overlapping windows will re-propose the same edge — dedupe by
    ``(source, target)``, keeping the higher ``edge_prob``, when stitching).

    Returns ``{"nodes": {node_id: (t, z, y, x)}, "edges": [(u, v, edge_prob), ...]}``
    with ``node_id = (t, local_index)``.
    """
    T, N = mask.shape
    nodes: dict[tuple[int, int], tuple[int, float, float, float]] = {}
    for t in range(T):
        for i in range(N):
            if mask[t, i]:
                z, y, x = coords[t, i].tolist()
                nodes[(t, i)] = (t_start + t, z, y, x)

    edges: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for (t_a, t_b), logits in edge_logits.items():
        probs = torch.sigmoid(logits)
        for i in range(N):
            if not mask[t_a, i]:
                continue
            for j in range(N):
                if not mask[t_b, j] or logits[i, j] == float("-inf"):
                    continue
                if logits[i, j] > threshold:
                    edges.append(((t_a, i), (t_b, j), float(probs[i, j])))

    return {"nodes": nodes, "edges": edges}
