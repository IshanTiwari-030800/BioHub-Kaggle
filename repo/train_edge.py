#!/usr/bin/env python
"""Train the NodeTransformer edge scorer on top of a pretrained TemporalUNet3D detector.

Stage (2) of the pipeline (see ``README.md`` / ``NODE_TRANSFORMER.md``): given
per-frame GT centroids over a WINDOW_SIZE=3 window, predicts the probability
that candidate i in frame t_a and candidate j in frame t_b are the same cell,
for every pair of frames in the window (both consecutive and skip-frame).
Follows the same paradigm as ``train.py`` (which trains the detector this
module builds on), extended from pilkwang's vendored T=2 edge-predictor
recipe (dense cross-attention, ``train_unet_transformer.py`` in the
``biohub-tracking-support-pack-50ep-v1`` kagglehub cache) to T=3 with the
deformable-attention backbone already trained in this repo.

Trains on GT centroids directly (teacher forcing), not the detector's own
noisy peaks — unlike pilkwang's detect-and-match training loop. This is a
deliberate simplification (see NODE_TRANSFORMER.md "Known limitations"): it's
simpler and more robust to get working, at the cost of a train/inference
distribution mismatch (inference still runs on peak-extracted detections).
The backbone is fine-tuned jointly with the edge scorer by default
(``freeze_backbone=False``), matching pilkwang's joint end-to-end training
rather than a frozen-feature approach.

Multi-GPU training uses DistributedDataParallel: the script self-spawns one
process per GPU (no need to launch via ``torchrun``), each holding a full
model replica and its own shard of the data (via ``DistributedSampler``).
Metrics are logged to Weights & Biases from rank 0 only. Both the best
(highest val edge-F1) and the most recent epoch's weights are checkpointed
to <output_dir>/edge_model_{best,last}.pt.

Data layout expected (see BioHub-Kaggle/README.md):
    <data_dir>/<id>.zarr  — image volume (T, Z, Y, X), uint16
    <data_dir>/<id>.geff  — ground-truth graph, node attrs t, z, y, x, real lineage edges

This module has no CLI: build an ``EdgeTrainConfig`` and call ``run(config)``,
e.g. from a Kaggle notebook cell:

    from train_edge import EdgeTrainConfig, run
    run(EdgeTrainConfig(
        data_dir="/kaggle/input/.../train",
        unet_weights="/kaggle/input/.../centroid_unet_best.pt",
        n_epochs=50, n_gpus=2,
    ))
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from itertools import cycle as _cycle
from pathlib import Path

import geff
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb
import zarr
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir / "models"))
sys.path.insert(0, str(_repo_dir))
from temporal_unet import TemporalUNet3D  # noqa: E402
from node_transformer import NodeTransformer, edge_focal_loss, edge_precision_recall  # noqa: E402

from datasets import (  # noqa: E402
    WINDOW_SIZE,
    VideoMeta,
    discover_datasets,
    parse_scale,
)
from train import (  # noqa: E402
    DEFAULT_AUGMENTATIONS,
    CentroidUNet,
    _dataloader_worker_init_fn,
    _default_data_dir,
    centroid_recall,
    detection_loss,
    is_main_process,
    reduce_sum,
    setup_distributed,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Every unordered frame-pair the model scores. For WINDOW_SIZE=3 this is
# exactly {(0,1), (1,2), (0,2)} — i.e. both consecutive pairs and the
# skip-frame pair — since NodeTransformer(skip_frame_edges=True) scores
# consecutive pairs plus every t -> t+2 pair.
_PAIRS: list[tuple[int, int]] = list(itertools.combinations(range(WINDOW_SIZE), 2))


# =============================================================================
# Config
# =============================================================================

@dataclass
class EdgeTrainConfig:
    """All training hyperparameters. Build one in your notebook and pass it to ``run``."""

    data_dir: str | Path | None = None
    output_dir: str | Path | None = None
    unet_weights: str | Path | None = None  # pretrained CentroidUNet checkpoint (required)

    n_epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 4  # per-GPU
    num_workers: int = 4

    unet_out_channels: int = 32
    unet_layers: list[int] = field(default_factory=lambda: [32, 64, 128])
    unet_n_heads: int = 4
    unet_n_points: int = 4
    freeze_backbone: bool = False  # False = fine-tune backbone jointly, matching pilkwang's recipe
    det_loss_weight: float = 0.1   # weight of the auxiliary detection loss when not frozen
    det_neg_weight: float = 0.05

    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ffn_dim: int = 256
    dropout: float = 0.1
    n_fourier_bands: int = 8
    max_link_distance_um: float = 14.0
    focal_gamma: float = 2.0

    downsample: tuple[int, int, int] = (1, 4, 4)  # (Z, Y, X) strides -> isotropic voxels
    val_frac: float = 0.15
    window_stride: int = 1
    max_iters: int | None = None
    pool_kernel: int = 3
    max_match_distance_um: float = 5.0
    seed: int = 0

    n_gpus: int | None = None  # None = all visible GPUs

    wandb_project: str = "biohub-cell-tracking"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"


# =============================================================================
# Data
# =============================================================================

@dataclass(frozen=True)
class EdgeFrameWindow:
    t_start: int
    coords: list[np.ndarray]  # WINDOW_SIZE arrays, each (N_i, 3) float32, downsampled voxel space
    pos_pairs: dict[tuple[int, int], np.ndarray]  # {(i,j): (K,2) int64} local indices into coords[i]/coords[j]


def load_edge_video_windows(
    zarr_path: Path,
    geff_path: Path,
    downsample: tuple[int, int, int],
    window_stride: int = 1,
) -> tuple[VideoMeta, list[EdgeFrameWindow]]:
    """Like ``train.load_video_windows``, but also carries GT edge indices per window.

    A GT edge (u, v) becomes a positive training pair for window offsets
    (i, j) whenever ``t[u] == t_start + i`` and ``t[v] == t_start + j`` for
    some scored pair (i, j) in ``_PAIRS`` — this naturally covers both
    consecutive-frame lineage edges and genuine gap edges (annotator skipped
    a frame for that cell), since ``.geff`` only stores real graph edges, not
    a transitive closure.
    """
    group = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(group.attrs)
    raw_shape = tuple(group["0"].shape)  # (T, Z, Y, X)

    dz, dy, dx = downsample
    ds_shape = (
        raw_shape[0],
        -(-raw_shape[1] // dz),
        -(-raw_shape[2] // dy),
        -(-raw_shape[3] // dx),
    )

    scale = parse_scale(attrs)
    voxel_size = tuple(s * d for s, d in zip(scale, downsample))

    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})
    if "0.001" not in quantiles or "0.999" not in quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {zarr_path}")

    video_meta = VideoMeta(
        zarr_path=zarr_path,
        image_shape=ds_shape,
        downsample=downsample,
        voxel_size=voxel_size,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
    )

    graph, _ = geff.read(str(geff_path), node_props=["t", "z", "y", "x"])
    ds_arr = np.array(downsample, dtype=np.float32)

    node_t: dict[int, int] = {}
    per_frame_ids: dict[int, list[int]] = {}
    per_frame_coords: dict[int, list[tuple[float, float, float]]] = {}
    for nid, d in graph.nodes(data=True):
        t = int(d["t"])
        node_t[nid] = t
        per_frame_ids.setdefault(t, []).append(nid)
        per_frame_coords.setdefault(t, []).append(
            (d["z"] / ds_arr[0], d["y"] / ds_arr[1], d["x"] / ds_arr[2])
        )

    edges_by_source_t: dict[int, list[tuple[int, int, int]]] = {}  # t_a -> [(source_id, t_b, target_id)]
    for u, v in graph.edges():
        edges_by_source_t.setdefault(node_t[u], []).append((u, node_t[v], v))

    T = raw_shape[0]
    windows: list[EdgeFrameWindow] = []
    for t_start in range(0, T - WINDOW_SIZE + 1, window_stride):
        frame_ids = [per_frame_ids.get(t_start + i, []) for i in range(WINDOW_SIZE)]
        if any(len(ids) == 0 for ids in frame_ids):
            continue  # skip windows where any frame has zero annotated centroids

        frame_coords = [np.array(per_frame_coords[t_start + i], dtype=np.float32) for i in range(WINDOW_SIZE)]
        id_to_idx = [{nid: idx for idx, nid in enumerate(ids)} for ids in frame_ids]

        pos_pairs: dict[tuple[int, int], np.ndarray] = {}
        for i, j in _PAIRS:
            t_a, t_b = t_start + i, t_start + j
            rows = [
                (id_to_idx[i][source_id], id_to_idx[j][target_id])
                for source_id, t_target, target_id in edges_by_source_t.get(t_a, [])
                if t_target == t_b and source_id in id_to_idx[i] and target_id in id_to_idx[j]
            ]
            pos_pairs[(i, j)] = np.array(rows, dtype=np.int64).reshape(-1, 2)

        windows.append(EdgeFrameWindow(t_start=t_start, coords=frame_coords, pos_pairs=pos_pairs))

    return video_meta, windows


class EdgeWindowDataset(Dataset):
    """Windows of WINDOW_SIZE consecutive frames + GT centroids + GT edge indices.

    Images are read from zarr on-demand; the same augmentations ``train.py``
    uses apply here unchanged — they only ever transform coordinate *values*
    or flip the image, never reorder the node axis, so index-based
    ``pos_pairs`` stay valid after augmentation.
    """

    def __init__(
        self,
        video_data: list[tuple[VideoMeta, list[EdgeFrameWindow]]],
        augmentations: list | None = None,
    ):
        self._index: list[tuple[EdgeFrameWindow, VideoMeta]] = [
            (w, vm) for vm, windows in video_data for w in windows
        ]
        self.augmentations = augmentations or []

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        window, vm = self._index[idx]
        dz, dy, dx = vm.downsample
        target_shape = vm.image_shape[1:]

        z = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
        raw = z[window.t_start: window.t_start + WINDOW_SIZE, ::dz, ::dy, ::dx].astype(np.float32)
        imgs = torch.from_numpy((raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)).clamp(min=0.0)

        if list(imgs.shape[1:]) != list(target_shape):
            imgs = F.interpolate(
                imgs[:, None], size=target_shape, mode="trilinear", align_corners=False,
            )[:, 0]

        max_n = max(max(len(c) for c in window.coords), 1)
        coords = torch.zeros(WINDOW_SIZE, max_n, 3, dtype=torch.float32)
        mask = torch.zeros(WINDOW_SIZE, max_n, dtype=torch.bool)
        for i, c in enumerate(window.coords):
            n = len(c)
            if n:
                coords[i, :n] = torch.from_numpy(c)
                mask[i, :n] = True

        if self.augmentations:
            rng = np.random.default_rng()
            for aug in self.augmentations:
                imgs, coords, mask = aug(imgs, coords, mask, rng=rng)

        pos_masks = {}
        for (i, j), pairs in window.pos_pairs.items():
            m = torch.zeros(max_n, max_n, dtype=torch.bool)
            if len(pairs):
                m[pairs[:, 0], pairs[:, 1]] = True
            pos_masks[(i, j)] = m

        return {"imgs": imgs, "coords": coords, "mask": mask, "pos_masks": pos_masks}


def collate_edge_windows(batch: list[dict]) -> dict:
    """Pad variable node counts to the batch max (coords/mask/pos_masks alike)."""
    B = len(batch)
    W = batch[0]["imgs"].shape[0]
    max_nodes = max(item["coords"].shape[1] for item in batch)

    imgs = torch.stack([item["imgs"] for item in batch], dim=0)
    coords = torch.zeros(B, W, max_nodes, 3, dtype=torch.float32)
    mask = torch.zeros(B, W, max_nodes, dtype=torch.bool)
    for b, item in enumerate(batch):
        n = item["coords"].shape[1]
        coords[b, :, :n] = item["coords"]
        mask[b, :, :n] = item["mask"]

    pos_masks: dict[tuple[int, int], torch.Tensor] = {}
    for pair in _PAIRS:
        pm = torch.zeros(B, max_nodes, max_nodes, dtype=torch.bool)
        for b, item in enumerate(batch):
            src_n, tgt_n = item["pos_masks"][pair].shape
            pm[b, :src_n, :tgt_n] = item["pos_masks"][pair]
        pos_masks[pair] = pm

    return {"imgs": imgs, "coords": coords, "mask": mask, "pos_masks": pos_masks}


# =============================================================================
# Model
# =============================================================================

class EdgeModel(nn.Module):
    """``CentroidUNet`` backbone (pretrained) + ``NodeTransformer`` edge scorer.

    When ``freeze_backbone`` the backbone runs under ``no_grad`` (no
    detection loss computed either, since there's nothing to train it with);
    otherwise both the backbone and detection head keep training jointly
    with the edge scorer, matching pilkwang's end-to-end recipe.
    """

    def __init__(self, centroid_unet: CentroidUNet, node_transformer: NodeTransformer, freeze_backbone: bool):
        super().__init__()
        self.centroid_unet = centroid_unet
        self.node_transformer = node_transformer
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.centroid_unet.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        imgs: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        voxel_size: tuple[float, float, float],
    ) -> tuple[torch.Tensor | None, dict]:
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.centroid_unet.unet(imgs.unsqueeze(2))
            det_logits = None
        else:
            feats = self.centroid_unet.unet(imgs.unsqueeze(2))
            B, W = feats.shape[:2]
            det_logits = self.centroid_unet.detect_head(feats.reshape(B * W, *feats.shape[2:]))
            det_logits = det_logits.reshape(B, W, *det_logits.shape[1:])

        out = self.node_transformer(feats, coords, mask, voxel_size)
        return det_logits, out


def compute_total_loss(
    det_logits: torch.Tensor | None,
    out: dict,
    coords: torch.Tensor,
    mask: torch.Tensor,
    pos_masks: dict[tuple[int, int], torch.Tensor],
    config: EdgeTrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, edge_loss, det_loss)."""
    edge_losses = [
        edge_focal_loss(logits, pos_masks[pair], out["valid_masks"][pair], gamma=config.focal_gamma)
        for pair, logits in out["edge_logits"].items()
    ]
    edge_loss = sum(edge_losses) / len(edge_losses)

    det_loss = edge_loss.new_zeros(())
    if det_logits is not None:
        W = det_logits.shape[1]
        det_loss = sum(
            detection_loss(det_logits[:, i], coords[:, i], mask[:, i], config.det_neg_weight) for i in range(W)
        ) / W

    return edge_loss + config.det_loss_weight * det_loss, edge_loss, det_loss


# =============================================================================
# Training / evaluation loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: EdgeTrainConfig,
    voxel_size: tuple[float, float, float],
    rank: int,
    max_iters: int | None = None,
    global_step: int = 0,
    log_wandb: bool = False,
) -> tuple[float, float, float, int, int]:
    """Runs one local (per-rank) epoch. Returns (loss, edge_loss, det_loss, n_samples, global_step)."""
    model.train()
    loss_sum, edge_loss_sum, det_loss_sum, n = 0.0, 0.0, 0.0, 0

    if max_iters is not None:
        batch_iter, steps = _cycle(loader), max_iters
    else:
        batch_iter, steps = iter(loader), len(loader)

    pbar = tqdm(range(steps), desc="  train", leave=False, disable=not is_main_process(rank))
    for _ in pbar:
        batch = next(batch_iter)
        imgs = batch["imgs"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pos_masks = {k: v.to(device, non_blocking=True) for k, v in batch["pos_masks"].items()}
        B = imgs.shape[0]

        det_logits, out = model(imgs, coords, mask, voxel_size)
        loss, edge_loss, det_loss = compute_total_loss(det_logits, out, coords, mask, pos_masks, config)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item() * B
        edge_loss_sum += edge_loss.item() * B
        det_loss_sum += det_loss.item() * B
        n += B
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", edge=f"{edge_loss.item():.4f}")

        if log_wandb:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/edge_loss_step": edge_loss.item(),
                "train/det_loss_step": det_loss.item(),
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

    return loss_sum, edge_loss_sum, det_loss_sum, n, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: EdgeTrainConfig,
    voxel_size: tuple[float, float, float],
    rank: int,
) -> dict[str, float]:
    """Returns local (per-rank) sums; caller reduces across ranks before dividing."""
    model.eval()
    loss_sum, edge_loss_sum, det_loss_sum, n = 0.0, 0.0, 0.0, 0
    tp, fp, fn = 0, 0, 0
    matched, total_gt = 0, 0

    for batch in tqdm(loader, desc="  val", leave=False, disable=not is_main_process(rank)):
        imgs = batch["imgs"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pos_masks = {k: v.to(device, non_blocking=True) for k, v in batch["pos_masks"].items()}
        B = imgs.shape[0]

        det_logits, out = model(imgs, coords, mask, voxel_size)
        loss, edge_loss, det_loss = compute_total_loss(det_logits, out, coords, mask, pos_masks, config)

        loss_sum += loss.item() * B
        edge_loss_sum += edge_loss.item() * B
        det_loss_sum += det_loss.item() * B
        n += B

        for pair, logits in out["edge_logits"].items():
            b_tp, b_fp, b_fn = edge_precision_recall(logits, pos_masks[pair], out["valid_masks"][pair])
            tp += b_tp
            fp += b_fp
            fn += b_fn

        if det_logits is not None:
            for i in range(det_logits.shape[1]):
                m, t = centroid_recall(
                    det_logits[:, i], coords[:, i], mask[:, i],
                    voxel_size, config.pool_kernel, config.max_match_distance_um,
                )
                matched += m
                total_gt += t

    return {
        "loss_sum": loss_sum, "edge_loss_sum": edge_loss_sum, "det_loss_sum": det_loss_sum, "n": n,
        "tp": tp, "fp": fp, "fn": fn, "matched": matched, "total_gt": total_gt,
    }


# =============================================================================
# Main
# =============================================================================

def train(rank: int, world_size: int, device: torch.device, config: EdgeTrainConfig) -> None:
    data_dir = Path(config.data_dir) if config.data_dir else _default_data_dir()
    output_dir = Path(config.output_dir) if config.output_dir else _REPO_ROOT / "weights"
    is_main = is_main_process(rank)

    if config.unet_weights is None:
        raise ValueError(
            "EdgeTrainConfig.unet_weights must point to a pretrained CentroidUNet checkpoint "
            "(e.g. weights/centroid_unet_best.pt) — this stage builds on the trained detector."
        )
    unet_weights = Path(config.unet_weights)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "edge_config.json").write_text(json.dumps({
            "unet_out_channels": config.unet_out_channels,
            "unet_layers": config.unet_layers,
            "unet_n_heads": config.unet_n_heads,
            "unet_n_points": config.unet_n_points,
            "embed_dim": config.embed_dim,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "ffn_dim": config.ffn_dim,
            "n_fourier_bands": config.n_fourier_bands,
            "max_link_distance_um": config.max_link_distance_um,
            "downsample": list(config.downsample),
            "window_size": WINDOW_SIZE,
        }, indent=2))

        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, mode=config.wandb_mode,
            config={
                "epochs": config.n_epochs, "lr": config.lr, "batch_size_per_gpu": config.batch_size,
                "effective_batch_size": config.batch_size * world_size, "world_size": world_size,
                "freeze_backbone": config.freeze_backbone, "det_loss_weight": config.det_loss_weight,
                "embed_dim": config.embed_dim, "n_heads": config.n_heads, "n_layers": config.n_layers,
                "max_link_distance_um": config.max_link_distance_um, "focal_gamma": config.focal_gamma,
                "downsample": config.downsample, "val_frac": config.val_frac,
                "window_stride": config.window_stride, "window_size": WINDOW_SIZE, "seed": config.seed,
            },
        )

    pairs = discover_datasets(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No (zarr, geff) pairs found in {data_dir}")
    if is_main:
        print(f"Found {len(pairs)} datasets in {data_dir}", flush=True)

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * config.val_frac)))
    val_idx = set(order[:n_val].tolist())
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]
    if is_main:
        print(f"Split: {len(train_pairs)} train videos, {len(val_pairs)} val videos", flush=True)

    def _load(pairs: list[tuple[Path, Path]], desc: str) -> tuple[list[tuple[VideoMeta, list[EdgeFrameWindow]]], int]:
        data, n_skipped = [], 0
        for zarr_path, geff_path in tqdm(pairs, desc=desc, disable=not is_main):
            try:
                data.append(load_edge_video_windows(zarr_path, geff_path, config.downsample, config.window_stride))
            except Exception as e:
                n_skipped += 1
                if is_main:
                    warnings.warn(f"Skipping unreadable dataset {zarr_path.stem}: {e}", stacklevel=2)
        n_windows = sum(len(w) for _, w in data)
        if is_main:
            print(f"  {desc}: {n_windows} windows ({n_skipped} datasets skipped)", flush=True)
        return data, n_windows

    train_data, n_train_windows = _load(train_pairs, "loading train")
    val_data, n_val_windows = _load(val_pairs, "loading val")

    if n_train_windows == 0:
        raise RuntimeError("No usable training windows (every frame in a window needs >=1 GT centroid).")

    if is_main:
        wandb.log({"data/n_train_windows": n_train_windows, "data/n_val_windows": n_val_windows}, step=0)

    train_ds = EdgeWindowDataset(train_data, augmentations=DEFAULT_AUGMENTATIONS)
    val_ds = EdgeWindowDataset(val_data)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed) \
        if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    g = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=config.num_workers, collate_fn=collate_edge_windows,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0, generator=g, worker_init_fn=_dataloader_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=config.num_workers, collate_fn=collate_edge_windows,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
    )

    if is_main:
        print(f"World size: {world_size} | device: {device}", flush=True)

    unet = TemporalUNet3D(
        in_channels=1, out_channels=config.unet_out_channels, layers=config.unet_layers,
        n_heads=config.unet_n_heads, n_points=config.unet_n_points,
    )
    centroid_unet = CentroidUNet(unet, feat_channels=config.unet_out_channels)
    state = torch.load(unet_weights, map_location="cpu", weights_only=True)
    centroid_unet.load_state_dict(state, strict=True)
    if is_main:
        print(f"  Loaded pretrained backbone from {unet_weights}", flush=True)

    node_transformer = NodeTransformer(
        feat_channels=config.unet_out_channels, embed_dim=config.embed_dim, n_heads=config.n_heads,
        n_layers=config.n_layers, ffn_dim=config.ffn_dim, dropout=config.dropout, window_size=WINDOW_SIZE,
        n_fourier_bands=config.n_fourier_bands, max_link_distance_um=config.max_link_distance_um,
        skip_frame_edges=True,
    )

    model: nn.Module = EdgeModel(centroid_unet, node_transformer, freeze_backbone=config.freeze_backbone).to(device)

    if world_size > 1:
        ddp_kwargs = {"device_ids": [rank], "output_device": rank} if device.type == "cuda" else {}
        model = DDP(model, static_graph=True, **ddp_kwargs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable parameters: {n_params:,} (freeze_backbone={config.freeze_backbone})", flush=True)
        wandb.log({"model/n_trainable_params": n_params}, step=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)

    voxel_size = next(iter(train_data + val_data))[0].voxel_size
    best_f1 = 0.0
    best_path = output_dir / "edge_model_best.pt"
    last_path = output_dir / "edge_model_last.pt"
    global_step = 0

    for epoch in range(config.n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.monotonic()
        tr_loss_sum, tr_edge_sum, tr_det_sum, tr_n, global_step = train_one_epoch(
            model, train_loader, optimizer, device, config, voxel_size, rank,
            max_iters=config.max_iters, global_step=global_step, log_wandb=is_main,
        )
        train_time = time.monotonic() - t0

        t0 = time.monotonic()
        val = evaluate(model, val_loader, device, config, voxel_size, rank)
        val_time = time.monotonic() - t0
        scheduler.step()

        tr_loss_sum = reduce_sum(tr_loss_sum, device, world_size)
        tr_edge_sum = reduce_sum(tr_edge_sum, device, world_size)
        tr_det_sum = reduce_sum(tr_det_sum, device, world_size)
        tr_n = reduce_sum(tr_n, device, world_size)
        for k in val:
            val[k] = reduce_sum(val[k], device, world_size)

        train_loss = tr_loss_sum / max(tr_n, 1)
        val_loss = val["loss_sum"] / max(val["n"], 1)
        val_edge_loss = val["edge_loss_sum"] / max(val["n"], 1)
        val_det_loss = val["det_loss_sum"] / max(val["n"], 1)

        precision = val["tp"] / max(val["tp"] + val["fp"], 1)
        recall = val["tp"] / max(val["tp"] + val["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        det_recall = val["matched"] / max(val["total_gt"], 1) if val["total_gt"] > 0 else None

        is_best = f1 >= best_f1
        if is_best:
            best_f1 = f1

        if is_main:
            raw_model = model.module if isinstance(model, DDP) else model
            torch.save(raw_model.state_dict(), last_path)
            if is_best:
                torch.save(raw_model.state_dict(), best_path)

            marker = "*" if is_best else " "
            det_str = f" | det_recall={det_recall:.4f}" if det_recall is not None else ""
            print(
                f"Epoch {epoch:3d}/{config.n_epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | edge_p={precision:.4f} r={recall:.4f} f1={f1:.4f} "
                f"| best_f1={best_f1:.4f} {marker}{det_str} | train={train_time:.1f}s val={val_time:.1f}s",
                flush=True,
            )
            log_dict = {
                "epoch": epoch,
                "train/loss_epoch": train_loss,
                "train/edge_loss_epoch": tr_edge_sum / max(tr_n, 1),
                "train/det_loss_epoch": tr_det_sum / max(tr_n, 1),
                "val/loss": val_loss,
                "val/edge_loss": val_edge_loss,
                "val/det_loss": val_det_loss,
                "val/edge_precision": precision,
                "val/edge_recall": recall,
                "val/edge_f1": f1,
                "val/best_edge_f1": best_f1,
                "lr": scheduler.get_last_lr()[0],
                "time/train_sec": train_time,
                "time/val_sec": val_time,
            }
            if det_recall is not None:
                log_dict["val/det_recall"] = det_recall
            wandb.log(log_dict, step=global_step)

    if is_main:
        print(f"\nBest edge F1: {best_f1:.4f} | best={best_path} | last={last_path}", flush=True)
        wandb.finish()


def _run_worker(rank: int, world_size: int, config: EdgeTrainConfig) -> None:
    if world_size > 1:
        device = setup_distributed(rank, world_size)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train(rank, world_size, device, config)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def run(config: EdgeTrainConfig) -> None:
    """Entry point: build an ``EdgeTrainConfig`` and call this from your notebook."""
    n_gpus = config.n_gpus if config.n_gpus is not None else torch.cuda.device_count()
    n_gpus = max(1, n_gpus)

    if n_gpus > 1:
        mp.spawn(_run_worker, args=(n_gpus, config), nprocs=n_gpus, join=True)
    else:
        _run_worker(0, 1, config)


if __name__ == "__main__":
    run(EdgeTrainConfig(unet_weights=str(_REPO_ROOT / "weights" / "centroid_unet_best.pt")))
