#!/usr/bin/env python
"""Sweep det_threshold / pool_kernel_um / edge threshold across the FULL
held-out validation set (the 22 videos never seen by either train.py or
train_edge.py, per their default seed=0/val_frac=0.15 split), instead of
the single 15-frame sample used in sweep_threshold.py / THRESHOLD_SWEEP.md.

Processes ONE VIDEO AT A TIME: build its window cache (encode+TTA, the
expensive part), sweep the full config grid against that cache, write its
results to a checkpoint file, then free the cache before moving to the next
video. This bounds peak memory to ~1 video's cache (~1.3GB) regardless of how
many videos are swept -- an earlier version that cached every video's windows
simultaneously needed ~29GB on this 7.5GB machine and was silently OOM-killed
after 4/22 videos. Per-video checkpointing also means a second interruption
only costs the in-flight video: re-running this script skips videos already
present in OUT_JSONL.

The grid is a targeted refinement around THRESHOLD_SWEEP.md's single-video
optimum (det_threshold=0.7, pool_kernel_um=12.0, threshold=0.7) rather than a
from-scratch exploratory search -- the open question this sweep answers is
whether that optimum generalizes across videos of wildly different density
(e.g. 44b6_144b256d's ~65k estimated true cells vs 6bba_372c8cb8's ~7k), not
re-discovering the region from zero.

Each video is truncated to the first MAX_FRAMES_PER_VIDEO frames (15,
matching the original single-video sample).

Usage:
    python sweep_threshold_val.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tracksdata as td
import zarr

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir / "models"))
sys.path.insert(0, str(_repo_dir))

from predict import load_model  # noqa: E402
from datasets import WINDOW_SIZE, load_video_meta, load_window_imgs, discover_datasets  # noqa: E402
from sweep_threshold import run_config, estimated_n_total, DEVICE, _encode  # noqa: E402

_REPO_ROOT = _repo_dir.parent

WEIGHTS = _REPO_ROOT / "weights" / "edge_model_best.pt"
DATA_DIR = _REPO_ROOT / "data" / "train"
MAX_FRAMES_PER_VIDEO = 15
SEED = 0
VAL_FRAC = 0.15
OUT_JSONL = _REPO_ROOT / "weights" / "predictions" / "threshold_sweep_val_per_video.jsonl"
OUT_STDOUT_MARKER = "SWEEP_DONE"

# Fixed grid, centered on THRESHOLD_SWEEP.md's single-video optimum.
DET_THRESHOLDS = [0.6, 0.7, 0.8]
POOL_KERNEL_UMS = [8.0, 10.0, 12.0, 14.0]
EDGE_THRESHOLDS = [0.5, 0.7]
GRID = [
    {"det_threshold": dt, "pool_kernel_um": pk, "threshold": th}
    for dt in DET_THRESHOLDS for pk in POOL_KERNEL_UMS for th in EDGE_THRESHOLDS
]


def get_val_pairs() -> list[tuple[Path, Path]]:
    pairs = discover_datasets(DATA_DIR)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * VAL_FRAC)))
    val_idx = set(order[:n_val].tolist())
    return [pairs[i] for i in sorted(val_idx)]


def build_cache_for_video(centroid_unet, zarr_path: Path, downsample, max_frames: int):
    vm = load_video_meta(zarr_path, downsample)
    W = WINDOW_SIZE
    T = min(vm.image_shape[0], max_frames)
    voxel_size = vm.voxel_size
    ds_arr = np.array(downsample, dtype=np.float32)

    window_starts = list(range(0, max(T - W + 1, 0)))
    cache = []
    for ws in window_starts:
        frame_indices = list(range(ws, ws + W))
        imgs = load_window_imgs(vm, ws).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feats, det_logits = _encode(centroid_unet, imgs)
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = _encode(centroid_unet, imgs_flip)
                det_logits = det_logits + det_flip.flip(dims)
            det_logits = det_logits / 4
        cache.append((frame_indices, feats.detach(), det_logits.detach()))
    return cache, voxel_size, ds_arr, W, T


def load_gt_for_video(geff_path: Path, max_frames: int):
    gt_result = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    gt_full = gt_result[0] if isinstance(gt_result, tuple) else gt_result
    attrs = gt_full.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t"])
    keep_ids = attrs.filter(pl.col("t") < max_frames)[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    return gt_full.filter(node_ids=keep_ids).subgraph()


def already_done() -> set[str]:
    if not OUT_JSONL.exists():
        return set()
    done = set()
    with open(OUT_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["video"])
            except (json.JSONDecodeError, KeyError):
                pass  # tolerate a truncated last line from a prior crash
    return done


def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    skip = already_done()
    if skip:
        print(f"Resuming: {len(skip)} videos already done, skipping: {sorted(skip)}", flush=True)

    centroid_unet, node_transformer, downsample = load_model(WEIGHTS, DEVICE)
    val_pairs = get_val_pairs()
    print(f"Validation set: {len(val_pairs)} videos, grid: {len(GRID)} configs/video", flush=True)

    t_start = time.monotonic()
    with open(OUT_JSONL, "a") as out_fh:
        for i, (zarr_path, geff_path) in enumerate(val_pairs, 1):
            name = zarr_path.stem
            if name in skip:
                continue

            t0 = time.monotonic()
            try:
                cache, voxel_size, ds_arr, W, T = build_cache_for_video(
                    centroid_unet, zarr_path, downsample, MAX_FRAMES_PER_VIDEO
                )
                gt = load_gt_for_video(geff_path, T)
                total_frames = zarr.open_group(str(zarr_path), mode="r")["0"].shape[0]
                n_total = estimated_n_total(geff_path, T, total_frames)
            except Exception as e:  # noqa: BLE001 -- one bad video shouldn't kill the run
                print(f"[{i}/{len(val_pairs)}] SKIPPED {name}: {type(e).__name__}: {e}", flush=True)
                continue

            cache_time = time.monotonic() - t0
            print(
                f"[{i}/{len(val_pairs)}] cached {name}: {len(cache)} windows, "
                f"gt_nodes={gt.num_nodes()}, n_total={n_total:.1f} ({cache_time:.1f}s)",
                flush=True,
            )

            if gt.num_nodes() == 0:
                # metrics.py's node-matching join can't infer a join-key dtype
                # from a 0-row ground-truth frame (polars SchemaError) -- this
                # window genuinely has no GT signal to score against (sparse
                # annotation timing for this video), so skip it rather than
                # crash the whole multi-hour sweep over one video's edge case.
                print(f"    SKIPPED (no GT nodes in first {MAX_FRAMES_PER_VIDEO} frames)\n", flush=True)
                del cache
                gc.collect()
                continue

            results = []
            t1 = time.monotonic()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for cfg in GRID:
                    try:
                        r = run_config(
                            node_transformer, cache, voxel_size, ds_arr, W, gt, n_total, **cfg,
                        )
                    except Exception as e:  # noqa: BLE001 -- one bad config shouldn't kill the run
                        print(f"    CONFIG FAILED {cfg}: {type(e).__name__}: {e}", flush=True)
                        r = {**cfg, "aborted": True, "error": str(e)}
                    results.append(r)
            config_time = time.monotonic() - t1
            print(
                f"    swept {len(GRID)} configs in {config_time:.1f}s "
                f"(best adj_edge_jaccard so far: "
                f"{max((r['adj_edge_jaccard'] for r in results if not r['aborted']), default=float('nan')):.4f})",
                flush=True,
            )

            out_fh.write(json.dumps({
                "video": name, "gt_nodes": gt.num_nodes(), "n_total": n_total,
                "cache_time_sec": cache_time, "config_time_sec": config_time,
                "results": results,
            }, default=str) + "\n")
            out_fh.flush()

            del cache
            gc.collect()

            elapsed = time.monotonic() - t_start
            print(f"    total elapsed: {elapsed:.1f}s ({elapsed / 3600:.2f}h)\n", flush=True)

    print(OUT_STDOUT_MARKER, flush=True)


if __name__ == "__main__":
    main()
