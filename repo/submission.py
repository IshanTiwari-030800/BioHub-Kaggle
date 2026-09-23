#!/usr/bin/env python
"""Produce submission.csv for the Biohub Cell Tracking Kaggle competition.

Full pipeline, per video: our stage 1-3 (repo/predict.py: TemporalUNet3D
detector -> NodeTransformer edge scorer -> tracksdata ILP) -> stage 4
(repo/graph_calibration.py: motion relink, gap closing, safe divisions,
short-track pruning, trajectory smoothing, ported from the reference
notebook per README.md section 4) -> submission.csv row schema (README.md
section 4.14, mirrored from nbs/submissions/submission_exp.ipynb).

Stage 1-3 uses the threshold-sweep's winning config (THRESHOLD_SWEEP.md):
det_threshold=0.7, pool_kernel_um=12.0, edge threshold=0.7, ILP on.

Usage (Kaggle):
    python submission.py
    # auto-detects /kaggle/input/competitions/biohub-cell-tracking-during-development/test,
    # falls back to /kaggle/input/biohub-cell-tracking-during-development/test,
    # override with BIOHUB_TEST_DIR env var.

Usage (local dry run, no real test set -- reuses data/train/*.zarr, ignores
the .geff GT files entirely):
    BIOHUB_TEST_DIR=data/train python submission.py --limit 2 --max-frames 15
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir / "models"))
sys.path.insert(0, str(_repo_dir))

from predict import (  # noqa: E402
    PredictConfig, load_model, predict_video, build_graph, solve_ilp,
)
from graph_calibration import (  # noqa: E402
    CalibrationConfig, calibrate_graph, calibrated_graph_to_rows,
)

_REPO_ROOT = _repo_dir.parent

COMPETITION = "biohub-cell-tracking-during-development"
_COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]

# Threshold-sweep winning config (THRESHOLD_SWEEP.md), superseding predict.py's
# pilkwang-derived defaults.
SWEEP_PREDICT_CONFIG = PredictConfig(
    det_threshold=0.7,
    pool_kernel_um=12.0,
    edge_activation="softmax",
    threshold=0.7,
    use_ilp=True,
    ilp_edge_weight=-1.0,
    ilp_appearance_weight=0.1,
    ilp_disappearance_weight=0.1,
    ilp_division_weight=1.0,
)


def _default_test_dir() -> Path:
    env = os.environ.get("BIOHUB_TEST_DIR", "").strip()
    if env:
        return Path(env)
    for cand in _COMP_DIR_CANDIDATES:
        if cand.exists():
            return cand / "test"
    # No Kaggle competition mount found (e.g. local dry run) -- caller must
    # pass BIOHUB_TEST_DIR explicitly; this fallback only exists so the
    # module is importable without one.
    return _COMP_DIR_CANDIDATES[0] / "test"


def _graph_to_plain(graph) -> tuple[dict[int, dict], list[dict]]:
    """tracksdata graph -> (nodes_by_id, raw_edges) plain dicts, per README section 4.2."""
    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes_by_id[node_id] = {"node_id": node_id, "t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}

    raw_edges: list[dict] = []
    if graph.num_edges() > 0:
        for row in graph.edge_attrs().iter_rows(named=True):
            prob = row.get("edge_prob")
            raw_edges.append({
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if prob is None else float(prob),
            })
    return nodes_by_id, raw_edges


def run_video(
    dataset: str,
    zarr_path: Path,
    centroid_unet,
    node_transformer,
    downsample: tuple[int, int, int],
    device: torch.device,
    predict_cfg: PredictConfig,
    calib_cfg: CalibrationConfig,
    max_frames: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Run the full stage1-3 + stage4 pipeline for one video. Returns (node_rows, edge_rows, stats)."""
    t0 = time.monotonic()
    coords, edges = predict_video(centroid_unet, node_transformer, zarr_path, device, predict_cfg, downsample, max_frames=max_frames)
    predict_sec = time.monotonic() - t0

    graph = build_graph(coords, edges)
    raw_pred_nodes = graph.num_nodes()
    if predict_cfg.use_ilp and graph.num_edges() > 0:
        graph = solve_ilp(graph, predict_cfg)

    nodes_by_id, raw_edges = _graph_to_plain(graph)
    calibrated_nodes, calibrated_edges, stage4_stats = calibrate_graph(nodes_by_id, raw_edges, calib_cfg, zarr_path=zarr_path)
    if not calibrated_nodes:
        raise AssertionError(f"{dataset}: stage-4 calibration removed every node")

    node_rows, edge_rows = calibrated_graph_to_rows(dataset, calibrated_nodes, calibrated_edges)

    stats = {
        "dataset": dataset,
        "stage13_nodes": raw_pred_nodes,
        "stage13_edges": graph.num_edges(),
        "final_nodes": len(calibrated_nodes),
        "final_edges": len(calibrated_edges),
        "predict_sec": round(predict_sec, 1),
        **stage4_stats,
    }
    return node_rows, edge_rows, stats


def run_submission(
    test_dir: Path,
    weights_path: Path,
    out_csv: Path,
    out_stats_csv: Path,
    limit: int | None = None,
    max_frames: int | None = None,
    predict_cfg: PredictConfig = SWEEP_PREDICT_CONFIG,
    calib_cfg: CalibrationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calib_cfg = calib_cfg or CalibrationConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centroid_unet, node_transformer, downsample = load_model(weights_path, device)

    zarr_paths = sorted(test_dir.glob("*.zarr"))
    if limit is not None:
        zarr_paths = zarr_paths[:limit]
    if not zarr_paths:
        raise FileNotFoundError(f"No .zarr videos found under {test_dir}")

    all_rows: list[dict] = []
    stats_rows: list[dict] = []
    row_id = 0
    for zarr_path in zarr_paths:
        dataset = zarr_path.stem
        node_rows, edge_rows, stats = run_video(
            dataset, zarr_path, centroid_unet, node_transformer, downsample, device,
            predict_cfg, calib_cfg, max_frames=max_frames,
        )
        for row in [*node_rows, *edge_rows]:
            row["id"] = row_id
            row_id += 1
        all_rows.extend(node_rows)
        all_rows.extend(edge_rows)
        stats_rows.append(stats)
        print(f"  {dataset}: stage1-3={stats['stage13_nodes']}n/{stats['stage13_edges']}e -> "
              f"final={stats['final_nodes']}n/{stats['final_edges']}e ({stats['predict_sec']}s)", flush=True)

    submission = pd.DataFrame(all_rows, columns=CSV_COLUMNS).set_index("id")
    submission.to_csv(out_csv)
    run_stats = pd.DataFrame(stats_rows)
    run_stats.to_csv(out_stats_csv, index=False)
    print(f"\nWrote {out_csv} ({len(submission)} rows) and {out_stats_csv}", flush=True)
    return submission, run_stats


def check_schema(submission: pd.DataFrame) -> bool:
    """Mirrors nbs/submissions/submission_exp.ipynb cell 22."""
    exp_cols = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    assert list(submission.columns) == exp_cols, submission.columns
    nodes = submission[submission.row_type == "node"]
    edges = submission[submission.row_type == "edge"]
    assert (nodes[["source_id", "target_id"]] == -1).all().all(), "node rows must have source_id=target_id=-1"
    assert (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all(), "edge rows must have node_id=t=z=y=x=-1"
    ok = True
    for ds, g in submission.groupby("dataset"):
        ids = set(g[g.row_type == "node"].node_id)
        e = g[g.row_type == "edge"]
        if not (set(e.source_id) | set(e.target_id)).issubset(ids):
            ok = False
            print("DANGLING EDGE in", ds)
    print(f"schema OK: {ok} | node rows: {len(nodes)} | edge rows: {len(edges)} | datasets: {submission.dataset.nunique()}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce submission.csv.")
    parser.add_argument("--test-dir", type=str, default=None, help="Default: auto-detect Kaggle test dir, or $BIOHUB_TEST_DIR.")
    parser.add_argument("--weights", type=str, default=None, help="Default: weights/edge_model_best.pt")
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--out-stats", type=str, default="run_stats.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos (dry-run/debug).")
    parser.add_argument("--max-frames", type=int, default=None, help="Only process the first N frames per video (dry-run/debug).")
    args = parser.parse_args()

    test_dir = Path(args.test_dir) if args.test_dir else _default_test_dir()
    weights = Path(args.weights) if args.weights else _REPO_ROOT / "weights" / "edge_model_best.pt"

    print(f"test_dir={test_dir} | weights={weights} | limit={args.limit} | max_frames={args.max_frames}", flush=True)
    submission, run_stats = run_submission(
        test_dir=test_dir, weights_path=weights,
        out_csv=Path(args.out), out_stats_csv=Path(args.out_stats),
        limit=args.limit, max_frames=args.max_frames,
    )
    check_schema(submission)


if __name__ == "__main__":
    main()
