#!/usr/bin/env python
"""Run TemporalUNet3D + NodeTransformer inference and export a .geff lineage graph.

Stage (3) of the pipeline (see ``README.md``): turns the trained detector
(``CentroidUNet``/``TemporalUNet3D``) and edge scorer (``NodeTransformer``)
into an actual per-video candidate graph, then an ILP tracker resolves it
into a final lineage graph. Detection (max-pool local-max peaks + flip TTA),
edge filtering (softmax-then-threshold, greedy children/parents caps), and
ILP post-processing (``tracksdata.solvers.ILPSolver``, same cost weights) are
a straight port of pilkwang's vendored ``predict_unet_transformer.py``
(``biohub-tracking-support-pack-50ep-v1`` kagglehub dataset). The only
differences from that reference script:

  * Backbone is the deformable-attention ``TemporalUNet3D`` (T=3) trained in
    this repo, not pilkwang's dense T=2 backbone.
  * Edge scoring is ``NodeTransformer``'s single joint self-attention forward
    pass over the whole 3-frame window (scores every consecutive pair *and*
    the t -> t+2 skip pair at once), replacing pilkwang's per-pair bidirectional
    cross-attention calls. Because skip edges exist, sliding windows use
    stride 1 (instead of pilkwang's stride ``W-1``) so every consecutive *and*
    skip pair gets covered by at least one window — pilkwang's stride trick
    was only a valid full-coverage shortcut for his consecutive-only, W=2
    case.
  * The per-pair greedy children/parents cap (used only when ``use_ilp`` is
    off) is applied independently per scored pair, exactly like pilkwang's
    single-pair loop. With skip edges this means a node's parent/child count
    is capped per *pair*, not globally across all pairs touching it -- the
    same scoping pilkwang's own loop has, just now visible because there is
    more than one pair per window. The ILP path does not have this gap: it
    optimizes over the whole candidate graph at once.

Usage:
    python predict.py --debug-video data/train/6bba_372c8cb8 --use-ilp
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td
import zarr

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir / "models"))
sys.path.insert(0, str(_repo_dir))

from temporal_unet import TemporalUNet3D  # noqa: E402
from node_transformer import NodeTransformer  # noqa: E402
from train import CentroidUNet, _default_data_dir  # noqa: E402
from datasets import WINDOW_SIZE, load_video_meta, load_window_imgs, parse_scale  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

# All frame-pairs NodeTransformer(skip_frame_edges=True) scores in one forward
# call: for WINDOW_SIZE=3 this is {(0,1), (1,2), (0,2)}.
_PAIRS: list[tuple[int, int]] = list(itertools.combinations(range(WINDOW_SIZE), 2))

_DEFAULT_EDGE_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "unet_n_heads": 4,
    "unet_n_points": 4,
    "embed_dim": 128,
    "n_heads": 4,
    "n_layers": 3,
    "ffn_dim": 256,
    "n_fourier_bands": 8,
    "max_link_distance_um": 14.0,
    "downsample": [1, 4, 4],
    "window_size": WINDOW_SIZE,
}


# =============================================================================
# Config (mirrors pilkwang's PredictConfig field-for-field)
# =============================================================================

@dataclass
class PredictConfig:
    """All hyperparameters that can affect prediction quality / score.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"softmax"`` (row-normalised
        over competing sources, i.e. "at most one true parent per target",
        matching training's ``edge_focal_loss``) or ``"sigmoid"``
        (independent per-edge scores).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """

    det_threshold: float = 0.99
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 3.0  # max-pool kernel size in um for detection peak extraction

    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5

    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # When ILP is enabled it handles parent/children constraints itself,
        # so greedy limits are left unconstrained (None). When ILP is
        # disabled, default to 1/2 to avoid unconstrained edge assignment.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2


# =============================================================================
# Model loading
# =============================================================================

def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[CentroidUNet, NodeTransformer, tuple[int, int, int]]:
    """Reconstruct CentroidUNet + NodeTransformer from a train_edge.py checkpoint.

    Reads ``edge_config.json`` from the same directory as the weights file
    (written by ``train_edge.py``'s ``train()``). Falls back to
    ``_DEFAULT_EDGE_CONFIG`` (``EdgeTrainConfig`` defaults) if missing.
    """
    config_path = weights_path.parent / "edge_config.json"
    if config_path.exists():
        config = {**_DEFAULT_EDGE_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: {config_path} not found, using EdgeTrainConfig defaults.", flush=True)
        config = _DEFAULT_EDGE_CONFIG

    unet = TemporalUNet3D(
        in_channels=1, out_channels=config["unet_out_channels"], layers=config["unet_layers"],
        n_heads=config["unet_n_heads"], n_points=config["unet_n_points"],
    )
    centroid_unet = CentroidUNet(unet, feat_channels=config["unet_out_channels"])
    node_transformer = NodeTransformer(
        feat_channels=config["unet_out_channels"], embed_dim=config["embed_dim"],
        n_heads=config["n_heads"], n_layers=config["n_layers"], ffn_dim=config["ffn_dim"],
        dropout=0.0, window_size=config["window_size"], n_fourier_bands=config["n_fourier_bands"],
        max_link_distance_um=config["max_link_distance_um"], skip_frame_edges=True,
    )

    state = torch.load(weights_path, map_location=device, weights_only=True)
    centroid_sd = {k[len("centroid_unet."):]: v for k, v in state.items() if k.startswith("centroid_unet.")}
    node_sd = {k[len("node_transformer."):]: v for k, v in state.items() if k.startswith("node_transformer.")}
    if not centroid_sd or not node_sd:
        raise ValueError(f"{weights_path} does not look like a train_edge.py EdgeModel checkpoint")

    centroid_unet.load_state_dict(centroid_sd, strict=True)
    node_transformer.load_state_dict(node_sd, strict=True)
    centroid_unet.to(device).eval()
    node_transformer.to(device).eval()

    downsample = tuple(config["downsample"])
    return centroid_unet, node_transformer, downsample


def _encode(centroid_unet: CentroidUNet, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(1, W, Z, Y, X) -> (feats (1,W,C,Z,Y,X), det_logits (1,W,1,Z,Y,X))."""
    feats = centroid_unet.unet(imgs.unsqueeze(2))
    B, W = feats.shape[:2]
    det_logits = centroid_unet.detect_head(feats.reshape(B * W, *feats.shape[2:]))
    det_logits = det_logits.reshape(B, W, *det_logits.shape[1:])
    return feats, det_logits


# =============================================================================
# Detection: max-pool local-max peak extraction (verbatim port of pilkwang's
# _detect_cells_pooled / pool_kernel_from_um)
# =============================================================================

def pool_kernel_from_um(um: float, voxel_size: tuple[float, ...]) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel."""
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _detect_cells_pooled(
    det_logits: torch.Tensor,  # (1, Z, Y, X)
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Returns (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def predict_video(
    centroid_unet: CentroidUNet,
    node_transformer: NodeTransformer,
    zarr_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    downsample: tuple[int, int, int],
    max_frames: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run inference on a single video using dense sliding windows of WINDOW_SIZE frames.

    Returns
    -------
    coords : np.ndarray, shape (N, 4) -- columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples, indices into coords.
    """
    vm = load_video_meta(zarr_path, downsample)
    W = WINDOW_SIZE
    T = vm.image_shape[0] if max_frames is None else min(vm.image_shape[0], max_frames)
    voxel_size = vm.voxel_size
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)
    ds_arr = np.array(downsample, dtype=np.float32)

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count = 0
    all_edges: list[tuple[int, int, float, float]] = []

    # Dense stride: unlike pilkwang's stride=W-1 (a full-coverage shortcut
    # valid only for consecutive-only, W=2 windows), skip-frame edges here
    # need every window offset so every (t,t+1) *and* (t,t+2) pair is scored
    # by at least one window.
    window_starts = list(range(0, max(T - W + 1, 0)))

    for ws in window_starts:
        frame_indices = list(range(ws, ws + W))
        imgs = load_window_imgs(vm, ws).unsqueeze(0).to(device)  # (1, W, Z, Y, X)

        feats, det_logits = _encode(centroid_unet, imgs)

        if cfg.det_tta:
            # Flip-Y, flip-X, flip-XY TTA; Z excluded (highly anisotropic axis).
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = _encode(centroid_unet, imgs_flip)
                det_logits = det_logits + det_flip.flip(dims)
            det_logits = det_logits / 4

        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(det_logits[0, f_idx], t, cfg.det_threshold, pool_k)
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )

        n_per_frame = [coord_offset[t][1] - coord_offset[t][0] for t in frame_indices]
        max_n = max(max(n_per_frame), 1)
        coords_t = torch.zeros(1, W, max_n, 3, dtype=torch.float32, device=device)
        mask_t = torch.zeros(1, W, max_n, dtype=torch.bool, device=device)
        local_idx: list[np.ndarray] = []
        for f_idx, t in enumerate(frame_indices):
            s, e = coord_offset[t]
            n = e - s
            local_idx.append(np.arange(s, e))
            if n:
                c = coords_so_far[s:e, 1:].astype(np.float32)
                coords_t[0, f_idx, :n] = torch.from_numpy(c)
                mask_t[0, f_idx, :n] = True

        out = node_transformer(feats, coords_t, mask_t, voxel_size)

        for (i, j), logits in out["edge_logits"].items():
            t_src, t_tgt = frame_indices[i], frame_indices[j]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            n_src, n_tgt = n_per_frame[i], n_per_frame[j]
            if n_src == 0 or n_tgt == 0:
                continue

            raw = logits[0, :n_src, :n_tgt]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0)
                probs = torch.nan_to_num(probs, nan=0.0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [
                    (probs[a, b], a, b)
                    for a in range(n_src)
                    for b in range(n_tgt)
                    if probs[a, b] > cfg.threshold
                ],
                reverse=True,
            )

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}
            for prob, a, b in candidates:
                n_ch = children_count.get(a, 0)
                n_pa = parents_count.get(b, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue

                gi, gj = int(local_idx[i][a]), int(local_idx[j][b])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32) - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[a] = n_ch + 1
                parents_count[b] = n_pa + 1

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr  # rescale spatial coords back to original resolution
    coords = coords.astype(np.int16)
    return coords, all_edges


# =============================================================================
# Graph building + ILP
# =============================================================================

def build_graph(coords: np.ndarray, edges: list[tuple[int, int, float, float]]) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detection coords and predicted edges."""
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)} for t, z, y, x in coords
    ])

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {"source_id": node_ids[src], "target_id": node_ids[tgt], "edge_prob": prob, "edge_dist": dist}
            for src, tgt, prob, dist in edges
        ])

    return graph


def solve_ilp(graph: td.graph.BaseGraph, cfg: PredictConfig) -> td.graph.BaseGraph:
    solver = td.solvers.ILPSolver(
        edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
        appearance_weight=cfg.ilp_appearance_weight,
        disappearance_weight=cfg.ilp_disappearance_weight,
        division_weight=cfg.ilp_division_weight,
    )
    return solver.solve(graph)


def save_graph(graph: td.graph.BaseGraph, output_path: Path) -> None:
    import shutil
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".geff":
        output_path = output_path.with_suffix(".geff")
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    graph.to_geff(output_path)


# =============================================================================
# CLI
# =============================================================================

def predict_one(
    name: str,
    data_dir: Path,
    weights: Path,
    out_dir: Path,
    cfg: PredictConfig,
    device: torch.device,
) -> dict:
    zarr_path = data_dir / f"{name}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    centroid_unet, node_transformer, downsample = load_model(weights, device)

    t0 = time.monotonic()
    coords, edges = predict_video(centroid_unet, node_transformer, zarr_path, device, cfg, downsample)
    predict_sec = time.monotonic() - t0

    graph = build_graph(coords, edges)
    n_candidate_edges = graph.num_edges()

    ilp_sec = 0.0
    if cfg.use_ilp and graph.num_edges() > 0:
        t0 = time.monotonic()
        graph = solve_ilp(graph, cfg)
        ilp_sec = time.monotonic() - t0

    out_path = out_dir / f"{name}.geff"
    save_graph(graph, out_path)

    n_div = sum(1 for _ in graph.dividing_nodes())
    n_frames = int(coords[:, 0].max()) + 1 if len(coords) else 0

    return {
        "name": name,
        "n_frames": n_frames,
        "n_nodes": graph.num_nodes(),
        "n_candidate_edges": n_candidate_edges,
        "n_final_edges": graph.num_edges(),
        "n_divisions": n_div,
        "used_ilp": cfg.use_ilp,
        "predict_sec": predict_sec,
        "ilp_sec": ilp_sec,
        "geff_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CentroidUNet + NodeTransformer + ILP prediction.")
    parser.add_argument("--data-dir", type=str, default=None, help="Default: _default_data_dir()")
    parser.add_argument("--debug-video", type=str, required=True,
                         help="Path to a single dataset, without extension, e.g. data/train/6bba_372c8cb8")
    parser.add_argument("--weights", type=str, default=None,
                         help="Default: weights/edge_model_best.pt")
    parser.add_argument("--out-dir", type=str, default=None, help="Default: weights/predictions")
    parser.add_argument("--det-threshold", type=float, default=0.99)
    parser.add_argument("--pool-kernel-um", type=float, default=3.0)
    parser.add_argument("--edge-activation", type=str, default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-ilp", action="store_true")
    parser.add_argument("--ilp-edge-weight", type=float, default=-1.0)
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.1)
    parser.add_argument("--ilp-disappearance-weight", type=float, default=0.1)
    parser.add_argument("--ilp-division-weight", type=float, default=1.0)
    args = parser.parse_args()

    debug_video = Path(args.debug_video)
    data_dir = Path(args.data_dir) if args.data_dir else debug_video.parent
    name = debug_video.name
    weights = Path(args.weights) if args.weights else _REPO_ROOT / "weights" / "edge_model_best.pt"
    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "weights" / "predictions"

    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        pool_kernel_um=args.pool_kernel_um,
        edge_activation=args.edge_activation,
        threshold=args.threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = predict_one(name, data_dir, weights, out_dir, cfg, device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
