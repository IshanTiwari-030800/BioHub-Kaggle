#!/usr/bin/env python
"""Sweep det_threshold / pool_kernel_um (and lightly, edge threshold) for
repo/predict.py's pipeline, scored with the real competition-proxy metric.

Caches the expensive TemporalUNet3D encode+TTA pass once per window (the
part that does NOT depend on det_threshold/pool_kernel_um), then replays
only the cheap steps (peak extraction, NodeTransformer forward, edge
assignment, ILP, scoring) per config. This is a read-only analysis script;
it does not modify predict.py/metrics.py/division_metrics.py.

Usage:
    python sweep_threshold.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tracksdata as td

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir / "models"))
sys.path.insert(0, str(_repo_dir))

from predict import (  # noqa: E402
    load_model, _encode, pool_kernel_from_um, _detect_cells_pooled,
    build_graph, solve_ilp, PredictConfig,
)
from datasets import WINDOW_SIZE, load_video_meta, load_window_imgs  # noqa: E402
from metrics import evaluate, node_recall, per_sample_metrics, EvaluationResult  # noqa: E402

_REPO_ROOT = _repo_dir.parent

DEVICE = torch.device("cpu")
WEIGHTS = _REPO_ROOT / "weights" / "edge_model_best.pt"
ZARR_PATH = _REPO_ROOT / "data" / "train" / "6bba_372c8cb8.zarr"
GEFF_PATH = _REPO_ROOT / "data" / "train" / "6bba_372c8cb8.geff"
MAX_FRAMES = 15
SCALE = (1.625, 0.40625, 0.40625)
MAX_DISTANCE = 7.0
MAX_NODES_GUARD = 4000  # abort a config if node count blows up (attention/ILP would be too slow)


def build_cache():
    centroid_unet, node_transformer, downsample = load_model(WEIGHTS, DEVICE)
    vm = load_video_meta(ZARR_PATH, downsample)
    W = WINDOW_SIZE
    T = min(vm.image_shape[0], MAX_FRAMES)
    voxel_size = vm.voxel_size
    ds_arr = np.array(downsample, dtype=np.float32)

    window_starts = list(range(0, max(T - W + 1, 0)))
    cache = []
    t0 = time.monotonic()
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
        print(f"  cached window {ws} ({time.monotonic()-t0:.1f}s elapsed)", flush=True)

    print(f"Cache build: {time.monotonic()-t0:.1f}s for {len(cache)} windows", flush=True)
    return node_transformer, cache, voxel_size, ds_arr, W


def load_gt():
    gt_result = td.graph.IndexedRXGraph.from_geff(str(GEFF_PATH))
    gt_full = gt_result[0] if isinstance(gt_result, tuple) else gt_result
    attrs = gt_full.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t"])
    keep_ids = attrs.filter(pl.col("t") < MAX_FRAMES)[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    return gt_full.filter(node_ids=keep_ids).subgraph()


def estimated_n_total(geff_path: Path, n_frames_scored: int, total_frames: int) -> float:
    """True (not sparsely-annotated) cell count, prorated to the scored frame range.

    GT annotation here is deliberately sparse -- most real cells aren't
    annotated -- so ``gt_graph.num_nodes()`` massively understates the true
    cell count and makes ``total_node_ratio`` (and therefore
    ``adj_edge_jaccard``) look far worse than it is. Pilkwang's own
    ``evaluate.py`` (``_read_estimated_n_total``) reads
    ``estimated_number_of_nodes`` from the GEFF metadata ``extra`` dict
    instead, for exactly this reason.
    """
    from geff import GeffMetadata
    meta = GeffMetadata.read(geff_path)
    full_video_n_total = float((meta.extra or {})["estimated_number_of_nodes"])
    return full_video_n_total * n_frames_scored / total_frames


def run_config(
    node_transformer, cache, voxel_size, ds_arr, W, gt, n_total: float,
    det_threshold: float, pool_kernel_um: float,
    edge_activation: str = "softmax", threshold: float = 0.5,
    use_ilp: bool = True, max_link_distance_um: float | None = None,
) -> dict:
    pool_k = pool_kernel_from_um(pool_kernel_um, voxel_size)
    cfg = PredictConfig(
        det_threshold=det_threshold, pool_kernel_um=pool_kernel_um,
        edge_activation=edge_activation, threshold=threshold, use_ilp=use_ilp,
    )

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count = 0
    all_edges: list[tuple[int, int, float, float]] = []
    aborted = False

    for frame_indices, feats, det_logits in cache:
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(det_logits[0, f_idx], t, det_threshold, pool_k)
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        if global_node_count > MAX_NODES_GUARD:
            aborted = True
            break

        coords_so_far = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)

        n_per_frame = [coord_offset[t][1] - coord_offset[t][0] for t in frame_indices]
        max_n = max(max(n_per_frame), 1)
        coords_t = torch.zeros(1, W, max_n, 3, dtype=torch.float32, device=DEVICE)
        mask_t = torch.zeros(1, W, max_n, dtype=torch.bool, device=DEVICE)
        local_idx: list[np.ndarray] = []
        for f_idx, t in enumerate(frame_indices):
            s, e = coord_offset[t]
            n = e - s
            local_idx.append(np.arange(s, e))
            if n:
                c = coords_so_far[s:e, 1:].astype(np.float32)
                coords_t[0, f_idx, :n] = torch.from_numpy(c)
                mask_t[0, f_idx, :n] = True

        with torch.no_grad():
            out = node_transformer(feats, coords_t, mask_t, voxel_size, max_link_distance_um=max_link_distance_um)

        for (i, j), logits in out["edge_logits"].items():
            t_src, t_tgt = frame_indices[i], frame_indices[j]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            n_src, n_tgt = n_per_frame[i], n_per_frame[j]
            if n_src == 0 or n_tgt == 0:
                continue

            raw = logits[0, :n_src, :n_tgt]
            if edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0)
                probs = torch.nan_to_num(probs, nan=0.0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [(probs[a, b], a, b) for a in range(n_src) for b in range(n_tgt) if probs[a, b] > threshold],
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

    result_base = {
        "det_threshold": det_threshold, "pool_kernel_um": pool_kernel_um,
        "edge_activation": edge_activation, "threshold": threshold, "n_total": n_total,
    }
    if aborted:
        return {**result_base, "n_pred_nodes": global_node_count, "aborted": True,
                "node_recall": float("nan"), "edge_jaccard": float("nan"),
                "total_node_ratio": float("nan"), "adj_edge_jaccard": float("nan")}

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)

    graph = build_graph(coords, all_edges)
    n_pred_nodes = graph.num_nodes()
    n_candidate_edges = graph.num_edges()
    if use_ilp and graph.num_edges() > 0:
        graph = solve_ilp(graph, cfg)
        # ILPSolver.solve() returns a GraphView, which doesn't support .copy()
        # (division_metrics._match_full needs it) -- detach to a concrete graph.
        graph = graph.detach()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if graph.num_edges() == 0 or graph.num_nodes() == 0:
            er = EvaluationResult(
                edge_tp=0, edge_fp=0, edge_fn=gt.num_edges(),
                division_tp=0, division_fp=0, division_fn=0,
                num_pred_nodes=n_pred_nodes,
            )
            recall = 0.0
        else:
            er = evaluate(graph, gt, scale=SCALE, max_distance=MAX_DISTANCE)
            recall = node_recall(graph, gt)
        m = per_sample_metrics(er, n_total=n_total, node_recall=recall)

    return {**result_base, "n_pred_nodes": n_pred_nodes, "n_candidate_edges": n_candidate_edges,
            "n_final_edges": graph.num_edges(), "aborted": False, **m}


def main():
    node_transformer, cache, voxel_size, ds_arr, W = build_cache()
    gt = load_gt()
    import zarr
    total_frames = zarr.open_group(str(ZARR_PATH), mode="r")["0"].shape[0]
    n_total = estimated_n_total(GEFF_PATH, MAX_FRAMES, total_frames)
    print(
        f"GT (t<{MAX_FRAMES}): {gt.num_nodes()} annotated nodes, {gt.num_edges()} edges | "
        f"estimated true n_total (prorated from {total_frames}-frame video): {n_total:.1f}",
        flush=True,
    )

    results = []

    # Stage 1: sweep det_threshold at baseline pool_kernel_um=3.0
    det_thresholds = [0.5, 0.7, 0.9, 0.99, 0.995, 0.999, 0.9999, 0.99999]
    for dt in det_thresholds:
        t0 = time.monotonic()
        r = run_config(node_transformer, cache, voxel_size, ds_arr, W, gt, n_total, det_threshold=dt, pool_kernel_um=3.0)
        r["stage"] = "det_threshold_sweep"
        r["elapsed_sec"] = time.monotonic() - t0
        results.append(r)
        print(json.dumps(r), flush=True)

    # Stage 2: sweep pool_kernel_um at the best det_threshold(s) from stage 1
    valid_stage1 = [r for r in results if not r["aborted"] and r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    valid_stage1.sort(key=lambda r: r["adj_edge_jaccard"], reverse=True)
    best_dts = [r["det_threshold"] for r in valid_stage1[:2]] if valid_stage1 else [0.99]

    pool_kernel_ums = [3.0, 5.0, 8.0, 12.0, 15.0]
    for dt in best_dts:
        for pk in pool_kernel_ums:
            if pk == 3.0:
                continue  # already covered in stage 1
            t0 = time.monotonic()
            r = run_config(node_transformer, cache, voxel_size, ds_arr, W, gt, n_total, det_threshold=dt, pool_kernel_um=pk)
            r["stage"] = "pool_kernel_sweep"
            r["elapsed_sec"] = time.monotonic() - t0
            results.append(r)
            print(json.dumps(r), flush=True)

    # Stage 3: light sweep of edge threshold at the best (det_threshold, pool_kernel_um) so far
    valid_all = [r for r in results if not r["aborted"] and r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    valid_all.sort(key=lambda r: r["adj_edge_jaccard"], reverse=True)
    if valid_all:
        best = valid_all[0]
        for th in [0.3, 0.5, 0.7, 0.9]:
            if th == 0.5:
                continue
            t0 = time.monotonic()
            r = run_config(
                node_transformer, cache, voxel_size, ds_arr, W, gt, n_total,
                det_threshold=best["det_threshold"], pool_kernel_um=best["pool_kernel_um"], threshold=th,
            )
            r["stage"] = "edge_threshold_sweep"
            r["elapsed_sec"] = time.monotonic() - t0
            results.append(r)
            print(json.dumps(r), flush=True)

    out_path = _REPO_ROOT / "weights" / "predictions" / "threshold_sweep_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
