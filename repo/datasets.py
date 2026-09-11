"""Shared video/window data loading for TemporalUNet3D, decoupled from training.

``VideoMeta`` + image loading/normalization are used by both training
(``train.py``, which additionally needs GT centroids from a ``.geff`` graph)
and inference (``InferenceWindowDataset`` below, which needs neither labels
nor augmentations — just sliding windows of frames to feed the model).

Data layout expected (see BioHub-Kaggle/README.md):
    <data_dir>/<id>.zarr — image volume (T, Z, Y, X), uint16
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)  # (Z, Y, X) microns
WINDOW_SIZE = 3  # t, t+1, t+2


@dataclass(frozen=True)
class VideoMeta:
    zarr_path: Path
    image_shape: tuple[int, int, int, int]   # (T, Z_ds, Y_ds, X_ds), downsampled
    downsample: tuple[int, int, int]
    voxel_size: tuple[float, float, float]   # physical size per downsampled voxel (Z, Y, X)
    q_low: float
    q_high: float


def parse_scale(attrs: dict) -> tuple[float, float, float]:
    if "multiscales" in attrs:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform["type"] != "scale":
            raise ValueError(f"Transform type is not 'scale': {transform}")
        return tuple(transform["scale"][-3:])
    return DEFAULT_SCALE


def discover_datasets(data_dir: Path) -> list[tuple[Path, Path]]:
    """Find (zarr_path, geff_path) pairs with both files present."""
    pairs = []
    for zarr_path in sorted(data_dir.glob("*.zarr")):
        geff_path = data_dir / f"{zarr_path.stem}.geff"
        if geff_path.exists():
            pairs.append((zarr_path, geff_path))
    return pairs


def load_video_meta(zarr_path: Path, downsample: tuple[int, int, int]) -> VideoMeta:
    """Read a zarr video's shape/scale/normalization stats (no image data read here)."""
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

    return VideoMeta(
        zarr_path=zarr_path,
        image_shape=ds_shape,
        downsample=downsample,
        voxel_size=voxel_size,
        q_low=float(quantiles["0.001"]),
        q_high=float(quantiles["0.999"]),
    )


def load_window_imgs(vm: VideoMeta, t_start: int) -> torch.Tensor:
    """Read+normalize WINDOW_SIZE consecutive frames starting at t_start. Shape (W, Z, Y, X)."""
    dz, dy, dx = vm.downsample
    target_shape = vm.image_shape[1:]

    z = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
    raw = z[t_start: t_start + WINDOW_SIZE, ::dz, ::dy, ::dx].astype(np.float32)
    imgs = torch.from_numpy((raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)).clamp(min=0.0)

    if list(imgs.shape[1:]) != list(target_shape):
        imgs = F.interpolate(
            imgs[:, None], size=target_shape, mode="trilinear", align_corners=False,
        )[:, 0]
    return imgs


class InferenceWindowDataset(Dataset):
    """Sliding WINDOW_SIZE-frame windows over one or more videos, no labels/augmentation.

    Each item covers frames ``[t_start, t_start + WINDOW_SIZE)`` of a video.
    ``window_stride=1`` (default) gives every frame coverage from multiple
    overlapping windows, which callers can average/stitch at inference time.
    """

    def __init__(self, videos: list[VideoMeta], window_stride: int = 1):
        self.videos = videos
        self._index: list[tuple[int, int]] = [
            (video_idx, t_start)
            for video_idx, vm in enumerate(videos)
            for t_start in range(0, vm.image_shape[0] - WINDOW_SIZE + 1, window_stride)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        video_idx, t_start = self._index[idx]
        vm = self.videos[video_idx]
        imgs = load_window_imgs(vm, t_start)
        return {"imgs": imgs, "video_idx": video_idx, "t_start": t_start}
