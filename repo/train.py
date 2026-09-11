#!/usr/bin/env python
"""Train the deformable TemporalUNet3D to detect cell centroids over 3-frame windows.

Extends pilkwang's UNet detection training (window_size=2, dense MHA temporal
attention) to window_size=3: the model is fed three consecutive frames
(t, t+1, t+2), pooled/reshaped to isotropic voxels, and trained to predict a
per-voxel centroid heatmap for each of the three frames independently, using
the deformable cross-time attention in ``models/temporal_unet.py`` to let
each frame's detection borrow context from the other two. No edge/tracking
head is trained here — this script only trains the detection backbone.

Multi-GPU training uses DistributedDataParallel: the script self-spawns one
process per GPU (no need to launch via ``torchrun``), each holding a full
model replica and its own shard of the data (via ``DistributedSampler``).
Metrics are logged to Weights & Biases from rank 0 only. Both the best
(highest val recall) and the most recent epoch's weights are checkpointed
to <output_dir>/centroid_unet_{best,last}.pth.

Data layout expected (see BioHub-Kaggle/README.md):
    <data_dir>/<id>.zarr  — image volume (T, Z, Y, X), uint16
    <data_dir>/<id>.geff  — ground-truth graph, node attrs t, z, y, x

This module has no CLI: build a ``TrainConfig`` and call ``run(config)``,
e.g. from a Kaggle notebook cell:

    from train import TrainConfig, run
    run(TrainConfig(data_dir="/kaggle/input/.../train", n_epochs=50, n_gpus=2))
"""

from __future__ import annotations

import json
import os
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

from datasets import (  # noqa: E402
    DEFAULT_SCALE,
    WINDOW_SIZE,
    VideoMeta,
    discover_datasets,
    parse_scale,
)


# =============================================================================
# Paths
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent
_KAGGLE_TRAIN = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")


def _default_data_dir() -> Path:
    env = os.environ.get("BIOHUB_DATA_DIR")
    if env:
        return Path(env)
    if _KAGGLE_TRAIN.exists():
        return _KAGGLE_TRAIN
    return _REPO_ROOT / "data" / "train"


# =============================================================================
# Distributed helpers
# =============================================================================

def setup_distributed(rank: int, world_size: int) -> torch.device:
    """Init the process group for this rank and return its device."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_sum(value: float, device: torch.device, world_size: int) -> float:
    """Sum a scalar across all ranks; no-op for single-process runs."""
    if world_size == 1:
        return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrainConfig:
    """All training hyperparameters. Build one in your notebook and pass it to ``run``."""

    data_dir: str | Path | None = None
    output_dir: str | Path | None = None

    n_epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 4  # per-GPU
    num_workers: int = 4

    unet_out_channels: int = 32
    unet_layers: list[int] = field(default_factory=lambda: [32, 64, 128])
    n_heads: int = 4
    n_points: int = 4
    unet_weights: str | Path | None = None  # pretrained UNet-only weights, loaded strict=False

    downsample: tuple[int, int, int] = (1, 4, 4)  # (Z, Y, X) strides -> isotropic voxels
    neg_weight: float = 0.05
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
class FrameWindow:
    t_start: int
    coords: list[np.ndarray]  # WINDOW_SIZE arrays, each (N_i, 3) float32, downsampled voxel space


def load_video_windows(
    zarr_path: Path,
    geff_path: Path,
    downsample: tuple[int, int, int],
    window_stride: int = 1,
) -> tuple[VideoMeta, list[FrameWindow]]:
    """Load per-window centroid metadata for one video (no image data read here)."""
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
    per_frame: dict[int, list[tuple[float, float, float]]] = {}
    for _, d in graph.nodes(data=True):
        per_frame.setdefault(int(d["t"]), []).append(
            (d["z"] / ds_arr[0], d["y"] / ds_arr[1], d["x"] / ds_arr[2])
        )

    T = raw_shape[0]
    windows: list[FrameWindow] = []
    for t_start in range(0, T - WINDOW_SIZE + 1, window_stride):
        frame_coords = [per_frame.get(t_start + i, []) for i in range(WINDOW_SIZE)]
        if any(len(c) == 0 for c in frame_coords):
            continue  # skip windows where any frame has zero annotated centroids
        windows.append(FrameWindow(
            t_start=t_start,
            coords=[np.array(c, dtype=np.float32) for c in frame_coords],
        ))

    return video_meta, windows


class CentroidWindowDataset(Dataset):
    """Windows of WINDOW_SIZE consecutive frames + their GT centroids.

    Images are read from zarr on-demand (no full video kept in RAM); a small
    set of augmentations is applied per-sample on the CPU.
    """

    def __init__(
        self,
        video_data: list[tuple[VideoMeta, list[FrameWindow]]],
        augmentations: list | None = None,
    ):
        self._index: list[tuple[FrameWindow, VideoMeta]] = [
            (w, vm) for vm, windows in video_data for w in windows
        ]
        self.augmentations = augmentations or []

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

        return {"imgs": imgs, "coords": coords, "mask": mask}


def collate_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable node counts to the batch max."""
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

    return {"imgs": imgs, "coords": coords, "mask": mask}


# =============================================================================
# Augmentations
# =============================================================================

def flip_augment(
    imgs: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, *, rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random flip along Z, Y, X independently (8 equally likely symmetries)."""
    flip = rng.random(3) < 0.5
    dims = [1 + ax for ax, f in enumerate(flip) if f]
    if not dims:
        return imgs, coords, mask

    imgs = imgs.flip(dims=dims)
    coords = coords.clone()
    spatial = imgs.shape[1:]  # (Z, Y, X)
    for ax in range(3):
        if flip[ax]:
            vals = coords[..., ax]
            vals[mask] = spatial[ax] - vals[mask] - 1
            coords[..., ax] = vals
    return imgs, coords, mask


def rot90_augment(
    imgs: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, *, rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random 90-degree rotation in the Y-X plane (rotation about the optical axis)."""
    k = int(rng.integers(0, 4))
    if k == 0:
        return imgs, coords, mask

    Y, X = imgs.shape[2], imgs.shape[3]
    imgs = torch.rot90(imgs, k, dims=(2, 3))

    coords = coords.clone()
    y, x = coords[..., 1].clone(), coords[..., 2].clone()
    if k == 1:
        new_y, new_x = (X - 1) - x, y
    elif k == 2:
        new_y, new_x = (Y - 1) - y, (X - 1) - x
    else:
        new_y, new_x = x, (Y - 1) - y
    coords[..., 1] = torch.where(mask, new_y, coords[..., 1])
    coords[..., 2] = torch.where(mask, new_x, coords[..., 2])
    return imgs, coords, mask


def brightness_augment(
    imgs: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, *,
    rng: np.random.Generator, shift_range: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random additive brightness shift."""
    shift = rng.uniform(-shift_range, shift_range)
    return imgs + shift, coords, mask


def contrast_gamma_augment(
    imgs: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, *,
    rng: np.random.Generator,
    contrast_range: tuple[float, float] = (0.8, 1.2),
    gamma_range: tuple[float, float] = (0.8, 1.25),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random multiplicative contrast + gamma jitter."""
    contrast = rng.uniform(*contrast_range)
    gamma = rng.uniform(*gamma_range)
    imgs = (imgs * contrast).clamp(min=0.0).pow(gamma)
    return imgs, coords, mask


def gaussian_noise_augment(
    imgs: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, *,
    rng: np.random.Generator, sigma_range: tuple[float, float] = (0.0, 0.03),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random additive Gaussian noise."""
    sigma = rng.uniform(*sigma_range)
    if sigma <= 0:
        return imgs, coords, mask
    noise = torch.from_numpy(rng.normal(0.0, sigma, size=tuple(imgs.shape)).astype(np.float32))
    return imgs + noise, coords, mask


DEFAULT_AUGMENTATIONS = [
    flip_augment,
    rot90_augment,
    brightness_augment,
    contrast_gamma_augment,
    gaussian_noise_augment,
]


# =============================================================================
# Model
# =============================================================================

class CentroidUNet(nn.Module):
    """TemporalUNet3D backbone + a 1x1x1 detection head predicting centroid heatmaps."""

    def __init__(self, unet: nn.Module, feat_channels: int):
        super().__init__()
        self.unet = unet
        self.detect_head = nn.Conv3d(feat_channels, 1, kernel_size=1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B, W, Z, Y, X) -> logits: (B, W, 1, Z, Y, X)
        feats = self.unet(imgs.unsqueeze(2))  # (B, W, C, Z, Y, X)
        B, W = feats.shape[:2]
        logits = self.detect_head(feats.reshape(B * W, *feats.shape[2:]))
        return logits.reshape(B, W, *logits.shape[1:])


# =============================================================================
# Loss + metrics
# =============================================================================

def detection_loss(
    logits: torch.Tensor,  # (B, 1, Z, Y, X)
    coords: torch.Tensor,  # (B, M, 3), downsampled voxel space
    mask: torch.Tensor,    # (B, M) bool
    neg_weight: float = 0.05,
) -> torch.Tensor:
    """BCE detection loss: GT centroid voxels are positive, all others lightly penalised."""
    B = logits.shape[0]
    spatial = logits.shape[2:]
    logits = logits[:, 0]  # (B, Z, Y, X)
    target = torch.zeros_like(logits)

    nt = mask.sum(dim=1).long()
    for b in range(B):
        n_gt = int(nt[b])
        if n_gt == 0:
            continue
        gt = coords[b, :n_gt]
        zi = gt[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt[:, 2].long().clamp(0, spatial[2] - 1)
        if len(torch.unique(torch.stack([zi, yi, xi], dim=1), dim=0)) < n_gt:
            warnings.warn(
                f"Sample {b}: some GT centroids collapsed to the same voxel after "
                "downsampling and are undetectable as separate peaks.", stacklevel=2,
            )
        target[b, zi, yi, xi] = 1.0

    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (target[0].numel() - n_pos).clamp(min=1)
    shape = (B,) + (1,) * len(spatial)
    weight = torch.where(
        target == 1.0, (1.0 / n_pos).reshape(shape), (neg_weight / n_neg).reshape(shape),
    )
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum") / B


@torch.no_grad()
def centroid_recall(
    logits: torch.Tensor,  # (B, 1, Z, Y, X)
    coords: torch.Tensor,  # (B, M, 3)
    mask: torch.Tensor,    # (B, M)
    voxel_size: tuple[float, float, float],
    pool_kernel: int = 3,
    max_match_distance_um: float = 5.0,
) -> tuple[int, int]:
    """Local-max peak detection -> greedy nearest-GT match. Returns (n_matched, n_gt)."""
    B, device = logits.shape[0], logits.device
    vs = torch.tensor(voxel_size, dtype=torch.float32, device=device)

    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pool_kernel // 2)
    is_peak = (logits == pooled) & (logits > 0.0)
    peak_idx = torch.nonzero(is_peak[:, 0])  # (N, 4): b, z, y, x

    matched, total = 0, 0
    for b in range(B):
        n_gt = int(mask[b].sum().item())
        if n_gt == 0:
            continue
        total += n_gt
        gt = coords[b, :n_gt] * vs
        det = peak_idx[peak_idx[:, 0] == b][:, 1:].float() * vs
        if det.shape[0] == 0:
            continue

        dists = torch.cdist(det, gt)
        min_d, min_i = dists.min(dim=1)
        order = min_d.argsort()
        taken = torch.zeros(n_gt, dtype=torch.bool, device=device)
        for idx in order:
            if min_d[idx] > max_match_distance_um:
                break
            gi = min_i[idx]
            if not taken[gi]:
                taken[gi] = True
                matched += 1

    return matched, total


# =============================================================================
# Training / evaluation loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    neg_weight: float,
    rank: int,
    max_iters: int | None = None,
    global_step: int = 0,
    log_wandb: bool = False,
) -> tuple[float, float, int]:
    """Runs one local (per-rank) epoch. Returns (loss_sum, n_samples, global_step)."""
    model.train()
    loss_sum, n = 0.0, 0

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
        B, W = imgs.shape[:2]

        logits = model(imgs)  # (B, W, 1, Z, Y, X)
        loss = sum(
            detection_loss(logits[:, i], coords[:, i], mask[:, i], neg_weight) for i in range(W)
        ) / W

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item() * B
        n += B
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if log_wandb:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/grad_norm": float(grad_norm),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

    return loss_sum, n, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    voxel_size: tuple[float, float, float],
    pool_kernel: int,
    max_match_distance_um: float,
    rank: int,
) -> tuple[float, int, int, int]:
    """Returns local (per-rank) (loss_sum, n_samples, n_matched, n_gt)."""
    model.eval()
    loss_sum, n = 0.0, 0
    matched, total_gt = 0, 0

    for batch in tqdm(loader, desc="  val", leave=False, disable=not is_main_process(rank)):
        imgs = batch["imgs"].to(device, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        B, W = imgs.shape[:2]

        logits = model(imgs)
        losses = []
        for i in range(W):
            losses.append(detection_loss(logits[:, i], coords[:, i], mask[:, i]))
            m, t = centroid_recall(
                logits[:, i], coords[:, i], mask[:, i], voxel_size, pool_kernel, max_match_distance_um,
            )
            matched += m
            total_gt += t
        loss_sum += (sum(losses) / W).item() * B
        n += B

    return loss_sum, n, matched, total_gt


# =============================================================================
# Main
# =============================================================================

def _dataloader_worker_init_fn(_worker_id: int) -> None:
    """Module-level (picklable) so it survives DataLoader workers started via spawn/forkserver."""
    np.random.seed(torch.initial_seed() % 2**32)


def train(rank: int, world_size: int, device: torch.device, config: TrainConfig) -> None:
    data_dir = Path(config.data_dir) if config.data_dir else _default_data_dir()
    output_dir = Path(config.output_dir) if config.output_dir else _REPO_ROOT / "weights"
    n_epochs = config.n_epochs
    lr = config.lr
    batch_size = config.batch_size
    num_workers = config.num_workers
    unet_out_channels = config.unet_out_channels
    unet_layers = list(config.unet_layers)
    n_heads = config.n_heads
    n_points = config.n_points
    unet_weights = Path(config.unet_weights) if config.unet_weights else None
    downsample = tuple(config.downsample)
    neg_weight = config.neg_weight
    val_frac = config.val_frac
    window_stride = config.window_stride
    max_iters = config.max_iters
    pool_kernel = config.pool_kernel
    max_match_distance_um = config.max_match_distance_um
    seed = config.seed
    wandb_project = config.wandb_project
    wandb_run_name = config.wandb_run_name
    wandb_entity = config.wandb_entity
    wandb_mode = config.wandb_mode

    is_main = is_main_process(rank)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps({
            "unet_out_channels": unet_out_channels,
            "unet_layers": unet_layers,
            "n_heads": n_heads,
            "n_points": n_points,
            "downsample": list(downsample),
            "window_size": WINDOW_SIZE,
        }, indent=2))

        wandb.init(
            project=wandb_project, entity=wandb_entity, name=wandb_run_name, mode=wandb_mode,
            config={
                "epochs": n_epochs, "lr": lr, "batch_size_per_gpu": batch_size,
                "effective_batch_size": batch_size * world_size, "world_size": world_size,
                "unet_out_channels": unet_out_channels, "unet_layers": unet_layers,
                "n_heads": n_heads, "n_points": n_points, "downsample": downsample,
                "neg_weight": neg_weight, "val_frac": val_frac, "window_stride": window_stride,
                "window_size": WINDOW_SIZE, "seed": seed,
            },
        )

    pairs = discover_datasets(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No (zarr, geff) pairs found in {data_dir}")
    if is_main:
        print(f"Found {len(pairs)} datasets in {data_dir}", flush=True)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val_idx = set(order[:n_val].tolist())
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]
    if is_main:
        print(f"Split: {len(train_pairs)} train videos, {len(val_pairs)} val videos", flush=True)

    def _load(pairs: list[tuple[Path, Path]], desc: str) -> list[tuple[VideoMeta, list[FrameWindow]]]:
        data = []
        n_skipped = 0
        for zarr_path, geff_path in tqdm(pairs, desc=desc, disable=not is_main):
            try:
                data.append(load_video_windows(zarr_path, geff_path, downsample, window_stride))
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

    train_ds = CentroidWindowDataset(train_data, augmentations=DEFAULT_AUGMENTATIONS)
    val_ds = CentroidWindowDataset(val_data)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed) \
        if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) \
        if world_size > 1 else None

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_windows,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0, generator=g, worker_init_fn=_dataloader_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_windows,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    if is_main:
        print(f"World size: {world_size} | device: {device}", flush=True)

    unet = TemporalUNet3D(
        in_channels=1, out_channels=unet_out_channels, layers=unet_layers,
        n_heads=n_heads, n_points=n_points,
    )
    if unet_weights is not None:
        state = torch.load(unet_weights, map_location="cpu", weights_only=True)
        missing, unexpected = unet.load_state_dict(state, strict=False)
        if is_main:
            print(f"  UNet weights: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)

    model: nn.Module = CentroidUNet(unet, feat_channels=unet_out_channels).to(device)

    if world_size > 1:
        ddp_kwargs = {"device_ids": [rank], "output_device": rank} if device.type == "cuda" else {}
        # static_graph=True: our forward pass is identical every step (no data-dependent
        # branching), which is required for gradient checkpointing to play safely with DDP.
        model = DDP(model, static_graph=True, **ddp_kwargs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Model parameters: {n_params:,}", flush=True)
        wandb.log({"model/n_params": n_params}, step=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    voxel_size = next(iter(train_data + val_data))[0].voxel_size
    best_recall = 0.0
    best_path = output_dir / "centroid_unet_best.pth"
    last_path = output_dir / "centroid_unet_last.pth"
    global_step = 0

    for epoch in range(n_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.monotonic()
        train_loss_sum, train_n, global_step = train_one_epoch(
            model, train_loader, optimizer, device, neg_weight, rank,
            max_iters=max_iters, global_step=global_step, log_wandb=is_main,
        )
        train_time = time.monotonic() - t0

        t0 = time.monotonic()
        val_loss_sum, val_n, matched, total_gt = evaluate(
            model, val_loader, device, voxel_size, pool_kernel, max_match_distance_um, rank,
        )
        val_time = time.monotonic() - t0
        scheduler.step()

        train_loss_sum = reduce_sum(train_loss_sum, device, world_size)
        train_n = reduce_sum(train_n, device, world_size)
        val_loss_sum = reduce_sum(val_loss_sum, device, world_size)
        val_n = reduce_sum(val_n, device, world_size)
        matched = reduce_sum(matched, device, world_size)
        total_gt = reduce_sum(total_gt, device, world_size)

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        val_recall = matched / max(total_gt, 1)

        is_best = val_recall >= best_recall
        if is_best:
            best_recall = val_recall

        if is_main:
            raw_model = model.module if isinstance(model, DDP) else model
            torch.save(raw_model.state_dict(), last_path)
            if is_best:
                torch.save(raw_model.state_dict(), best_path)

            marker = "*" if is_best else " "
            print(
                f"Epoch {epoch:3d}/{n_epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_recall={val_recall:.4f} | best={best_recall:.4f} {marker} | "
                f"train={train_time:.1f}s val={val_time:.1f}s",
                flush=True,
            )
            wandb.log({
                "epoch": epoch,
                "train/loss_epoch": train_loss,
                "val/loss": val_loss,
                "val/recall": val_recall,
                "val/best_recall": best_recall,
                "lr": scheduler.get_last_lr()[0],
                "time/train_sec": train_time,
                "time/val_sec": val_time,
            }, step=global_step)

    if is_main:
        print(f"\nBest recall: {best_recall:.4f} | best={best_path} | last={last_path}", flush=True)
        wandb.finish()


def _run_worker(rank: int, world_size: int, config: TrainConfig) -> None:
    if world_size > 1:
        device = setup_distributed(rank, world_size)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        train(rank, world_size, device, config)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def run(config: TrainConfig) -> None:
    """Entry point: build a ``TrainConfig`` and call this from your notebook."""
    n_gpus = config.n_gpus if config.n_gpus is not None else torch.cuda.device_count()
    n_gpus = max(1, n_gpus)

    if n_gpus > 1:
        mp.spawn(_run_worker, args=(n_gpus, config), nprocs=n_gpus, join=True)
    else:
        _run_worker(0, 1, config)


if __name__ == "__main__":
    run(TrainConfig())
