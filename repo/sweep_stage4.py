#!/usr/bin/env python
"""Sweep repo/graph_calibration.py's stage-4 params against our own pipeline's
output, scored with the real competition-proxy metric (repo/metrics.py).

Stage 1-3 (the expensive neural-net part) runs ONCE, using the threshold-sweep's
winning PredictConfig (THRESHOLD_SWEEP.md), to produce one raw ILP-solved
candidate graph. Every stage-4 config below is then just graph post-processing
scored against GT -- no re-running the backbone.

Usage: python sweep_stage4.py
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import tracksdata as td

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir / "models"))
sys.path.insert(0, str(_repo_dir))

from predict import PredictConfig, load_model, predict_video, build_graph, solve_ilp  # noqa: E402
from graph_calibration import CalibrationConfig, calibrate_graph  # noqa: E402
from metrics import evaluate, node_recall, per_sample_metrics  # noqa: E402
from geff import GeffMetadata  # noqa: E402

_REPO_ROOT = _repo_dir.parent
ZARR_PATH = _REPO_ROOT / "data" / "train" / "6bba_372c8cb8.zarr"
GEFF_PATH = _REPO_ROOT / "data" / "train" / "6bba_372c8cb8.geff"
MAX_FRAMES = 15
WEIGHTS = _REPO_ROOT / "weights" / "edge_model_best.pt"
SCALE = (1.625, 0.40625, 0.40625)
MAX_DISTANCE = 7.0

SWEEP_PREDICT_CONFIG = PredictConfig(
    det_threshold=0.7, pool_kernel_um=12.0, edge_activation="softmax", threshold=0.7,
    use_ilp=True, ilp_edge_weight=-1.0, ilp_appearance_weight=0.1,
    ilp_disappearance_weight=0.1, ilp_division_weight=1.0,
)


def build_base_graph():
    device = torch.device("cpu")
    centroid_unet, node_transformer, downsample = load_model(WEIGHTS, device)
    t0 = time.monotonic()
    coords, edges = predict_video(
        centroid_unet, node_transformer, ZARR_PATH, device, SWEEP_PREDICT_CONFIG,
        downsample, max_frames=MAX_FRAMES,
    )
    print(f"stage1-3: {time.monotonic()-t0:.1f}s | n_coords={len(coords)} n_edges={len(edges)}", flush=True)
    graph = build_graph(coords, edges)
    graph = solve_ilp(graph, SWEEP_PREDICT_CONFIG)

    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_by_id[nid] = {"node_id": nid, "t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    raw_edges: list[dict] = []
    if graph.num_edges() > 0:
        for row in graph.edge_attrs().iter_rows(named=True):
            prob = row.get("edge_prob")
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if prob is None else float(prob),
            })
    print(f"base ILP graph: {len(nodes_by_id)} nodes, {len(raw_edges)} edges", flush=True)
    return nodes_by_id, raw_edges


def plain_to_td_graph(nodes_by_id: dict[int, dict], edges: list[dict]) -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    ordered_ids = sorted(nodes_by_id.keys())
    td_ids = graph.bulk_add_nodes([
        {"t": int(nodes_by_id[nid]["t"]), "z": float(nodes_by_id[nid]["z"]),
         "y": float(nodes_by_id[nid]["y"]), "x": float(nodes_by_id[nid]["x"])}
        for nid in ordered_ids
    ])
    id_map = dict(zip(ordered_ids, td_ids))
    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {"source_id": id_map[int(e["source_id"])], "target_id": id_map[int(e["target_id"])],
             "edge_prob": float(e.get("edge_prob") or 0.0)}
            for e in edges if int(e["source_id"]) in id_map and int(e["target_id"]) in id_map
        ])
    return graph


def load_gt(max_frames: int):
    gt_result = td.graph.IndexedRXGraph.from_geff(str(GEFF_PATH))
    gt_full = gt_result[0] if isinstance(gt_result, tuple) else gt_result
    attrs = gt_full.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t"])
    keep_ids = attrs.filter(pl.col("t") < max_frames)[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    gt = gt_full.filter(node_ids=keep_ids).subgraph()

    import zarr
    total_frames = zarr.open_group(str(ZARR_PATH), mode="r")["0"].shape[0]
    meta = GeffMetadata.read(str(GEFF_PATH))
    est_total = float((meta.extra or {}).get("estimated_number_of_nodes"))
    n_total = est_total * max_frames / total_frames
    return gt, n_total


def score_config(nodes_by_id, raw_edges, cfg: CalibrationConfig, gt, n_total) -> dict:
    # calibrate_graph mutates nodes_by_id in place (e.g. inserts synthetic gap-close
    # nodes directly into the dict passed in) -- deep-copy so repeated sweep calls
    # each start from the same pristine base graph, not a progressively polluted one.
    nodes_by_id = copy.deepcopy(nodes_by_id)
    raw_edges = copy.deepcopy(raw_edges)
    calibrated_nodes, calibrated_edges, stats = calibrate_graph(nodes_by_id, raw_edges, cfg, zarr_path=ZARR_PATH)
    pred = plain_to_td_graph(calibrated_nodes, calibrated_edges)

    if pred.num_nodes() == 0 or pred.num_edges() == 0:
        return {"n_pred_nodes": pred.num_nodes(), "n_pred_edges": pred.num_edges(),
                "node_recall": 0.0, "edge_jaccard": float("nan"), "adj_edge_jaccard": 0.0,
                "total_node_ratio": float("nan"), "aborted": True}

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = evaluate(pred, gt, scale=SCALE, max_distance=MAX_DISTANCE)
        recall = node_recall(pred, gt)
    m = per_sample_metrics(er, n_total=n_total, node_recall=recall)
    m["n_pred_nodes"] = pred.num_nodes()
    m["n_pred_edges"] = pred.num_edges()
    m["aborted"] = False
    m["stats"] = stats
    return m


def fmt(cfg_desc: str, m: dict) -> str:
    return (f"{cfg_desc:55s} n_nodes={m['n_pred_nodes']:<5} n_edges={m['n_pred_edges']:<5} "
            f"recall={m['node_recall']:.3f} edge_j={m['edge_jaccard']:.3f} "
            f"ratio={m['total_node_ratio']:+.3f} adj_j={m['adj_edge_jaccard']:.4f} "
            f"div(tp/fp/fn)={m.get('division_tp')}/{m.get('division_fp')}/{m.get('division_fn')}")


def main():
    nodes_by_id, raw_edges = build_base_graph()
    gt, n_total = load_gt(MAX_FRAMES)
    print(f"GT (t<{MAX_FRAMES}): {gt.num_nodes()} nodes, {gt.num_edges()} edges | n_total (corrected)={n_total:.2f}\n", flush=True)

    results = []

    def run(stage: str, **overrides) -> dict:
        cfg = dataclasses.replace(CalibrationConfig(), **overrides)
        m = score_config(nodes_by_id, raw_edges, cfg, gt, n_total)
        m["stage"] = stage
        m["overrides"] = overrides
        results.append(m)
        print(fmt(f"{stage} {overrides}", m), flush=True)
        return m

    print("=== baseline (README-copied defaults, unchanged) ===", flush=True)
    baseline = run("baseline")

    print("\n=== stage A: motion relink on/off + learned_bonus ===", flush=True)
    run("motion_relink_off", output_motion_relink=False)
    for bonus in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        run("learned_bonus", motion_relink_learned_bonus=bonus)

    best_bonus_row = max([r for r in results if r["stage"] == "learned_bonus"], key=lambda r: r["adj_edge_jaccard"])
    best_bonus = best_bonus_row["overrides"]["motion_relink_learned_bonus"]
    print(f"  -> best learned_bonus so far: {best_bonus} (adj_j={best_bonus_row['adj_edge_jaccard']:.4f})", flush=True)

    print("\n=== stage A2: motion-relink tight/relaxed gates + velocity weight (at best learned_bonus) ===", flush=True)
    for tight, relaxed in [(4.0, 8.0), (6.0, 10.0), (8.0, 12.0), (10.0, 15.0), (12.0, 18.0)]:
        run("motion_gates", motion_relink_learned_bonus=best_bonus, motion_relink_tight_um=tight, motion_relink_relaxed_um=relaxed)
    for vw in [0.0, 0.3, 0.5, 0.7, 1.0]:
        run("velocity_weight", motion_relink_learned_bonus=best_bonus, motion_relink_velocity_weight=vw)

    stage_a_rows = [r for r in results if r["stage"] in ("motion_relink_off", "learned_bonus", "motion_gates", "velocity_weight", "baseline")]
    best_a = max(stage_a_rows, key=lambda r: r["adj_edge_jaccard"])
    best_a_overrides = dict(best_a["overrides"])
    print(f"\n  -> stage A winner: {best_a['stage']} {best_a_overrides} (adj_j={best_a['adj_edge_jaccard']:.4f})", flush=True)

    print("\n=== stage B: gap-close distance (at stage-A winner) ===", flush=True)
    for gap_um in [3.0, 4.4, 5.8, 7.0, 9.0, 11.6]:
        run("gap_close_um", **{**best_a_overrides, "gap_close_um": gap_um})
    run("gap_close_off", **{**best_a_overrides, "output_gap_close": False})

    stage_b_rows = [r for r in results if r["stage"] in ("gap_close_um", "gap_close_off")]
    best_b = max(stage_b_rows + [best_a], key=lambda r: r["adj_edge_jaccard"])
    best_b_overrides = dict(best_b["overrides"])
    print(f"\n  -> stage B winner: {best_b['stage']} {best_b_overrides} (adj_j={best_b['adj_edge_jaccard']:.4f})", flush=True)

    print("\n=== stage C: safe-division radii (CAVEAT: 0 real GT divisions in this sample) ===", flush=True)
    print(f"  GT divisions in this window: tp={baseline.get('division_tp')} fn={baseline.get('division_fn')} "
          f"-- division tuning here is not evidence-backed, sanity-check only.", flush=True)
    for parent_um in [3.0, 4.66, 6.0, 8.0]:
        run("safe_div_parent", **{**best_b_overrides, "safe_div_max_um": parent_um})
    run("safe_div_off", **{**best_b_overrides, "output_safe_divisions": False})

    print("\n=== stage D: short-track length + linefit smoothing (at stage-B winner) ===", flush=True)
    for min_len in [2, 4, 6, 8, 12]:
        run("min_track_len", **{**best_b_overrides, "output_min_track_len": min_len})
    for w in [0.0, 0.4, 0.8, 1.0]:
        run("linefit_weight", **{**best_b_overrides, "output_linefit_weight": w})

    for r in results:
        r.pop("stats", None)
    out_path = _REPO_ROOT / "weights" / "predictions" / "stage4_sweep_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {out_path}", flush=True)

    overall_best = max(results, key=lambda r: r["adj_edge_jaccard"])
    print(f"\n=== OVERALL BEST: {overall_best['stage']} {overall_best['overrides']} adj_j={overall_best['adj_edge_jaccard']:.4f} ===", flush=True)
    print(f"    baseline was adj_j={baseline['adj_edge_jaccard']:.4f}", flush=True)


if __name__ == "__main__":
    main()
