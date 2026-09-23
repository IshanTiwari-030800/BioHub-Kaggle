#!/usr/bin/env python
"""Diagnose why raw scores looked high and why the earlier adjusted score looked low.

Two independent checks against an existing prediction + GT pair (no new
inference needed):

1. Is tracksdata's node/edge matching gameable by spamming detections?
   (``check_matching_is_hungarian`` inspects ``tracksdata.metrics._matching``
   / ``_ctc_metrics`` directly to confirm the answer, then empirically
   buckets every predicted node by its nearest-GT distance and samples raw
   image intensity in each bucket to tell "duplicate detection" apart from
   "real but unannotated cell" apart from "background noise".)

2. Was ``per_sample_metrics``'s ``n_total`` computed correctly? Pilkwang's
   own ``evaluate.py`` (``_read_estimated_n_total``) reads
   ``estimated_number_of_nodes`` from the GEFF metadata ``extra`` dict —
   *not* ``gt_graph.num_nodes()`` — specifically because GT is sparsely
   annotated. Using the raw annotated count as ``n_total`` overstates
   "overprediction" whenever the true cell count is much larger than what's
   annotated (which is exactly this dataset).

Usage:
    python scoring_analysis.py --pred weights/predictions/6bba_372c8cb8_sample.geff \
        --gt data/train/6bba_372c8cb8.geff --zarr data/train/6bba_372c8cb8.zarr \
        --n-frames 15 --total-frames 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
import zarr
from scipy.spatial.distance import cdist

_repo_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_dir))

from metrics import evaluate, node_recall, per_sample_metrics  # noqa: E402

_ORIGINAL_SCALE = (1.625, 0.40625, 0.40625)  # (Z, Y, X) microns, this dataset's DEFAULT_SCALE


def _load_graph(path: Path):
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def nearest_gt_distance_buckets(
    pred_np: np.ndarray,  # (N, 4) t,z,y,x
    gt_np: np.ndarray,    # (M, 4) t,z,y,x
    n_frames: int,
    scale: tuple[float, float, float],
    max_distance: float,
) -> np.ndarray:
    """Per predicted node, distance (um) to the nearest GT node in the same frame."""
    scale_arr = np.array(scale)
    nearest = np.full(len(pred_np), np.inf)
    for t in range(n_frames):
        p_idx = np.where(pred_np[:, 0] == t)[0]
        g_idx = np.where(gt_np[:, 0] == t)[0]
        if len(p_idx) == 0 or len(g_idx) == 0:
            continue
        pc = pred_np[p_idx][:, 1:] * scale_arr
        gc = gt_np[g_idx][:, 1:] * scale_arr
        nearest[p_idx] = cdist(pc, gc).min(axis=1)
    return nearest


def sample_intensity(zarr_arr, t: int, z: int, y: int, x: int, r: int = 2) -> tuple[float, float]:
    Z, Y, X = zarr_arr.shape[1:]
    z0, z1 = max(0, z - r), min(Z, z + r + 1)
    y0, y1 = max(0, y - r), min(Y, y + r + 1)
    x0, x1 = max(0, x - r), min(X, x + r + 1)
    patch = zarr_arr[t, z0:z1, y0:y1, x0:x1]
    return float(patch.mean()), float(patch.max())


def intensity_report(name: str, coords: list[tuple[int, int, int, int]], zarr_arr) -> None:
    if not coords:
        print(f"  {name}: n=0")
        return
    means = [sample_intensity(zarr_arr, *c)[0] for c in coords]
    maxs = [sample_intensity(zarr_arr, *c)[1] for c in coords]
    print(
        f"  {name}: n={len(coords)} "
        f"median_mean_intensity={np.median(means):.1f} median_max_intensity={np.median(maxs):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--n-frames", type=int, required=True, help="Frames actually predicted on")
    parser.add_argument("--total-frames", type=int, required=True, help="Full video length (for prorating n_total)")
    parser.add_argument("--max-distance", type=float, default=7.0)
    args = parser.parse_args()

    pred = _load_graph(args.pred)
    gt_full = _load_graph(args.gt)

    pred_attrs = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    gt_attrs_full = gt_full.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    gt_attrs = gt_attrs_full.filter(pl.col("t") < args.n_frames)

    keep_ids = gt_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    gt = gt_full.filter(node_ids=keep_ids).subgraph()

    pred_np = pred_attrs.select(["t", "z", "y", "x"]).to_numpy()
    gt_np = gt_attrs.select(["t", "z", "y", "x"]).to_numpy()

    print(f"pred nodes={len(pred_np)}  gt nodes (t<{args.n_frames})={len(gt_np)}\n")

    # --- Part 1: is node_recall / edge_jaccard gameable by spamming? ---
    print("=== Part 1: distance-bucketed nearest-GT analysis ===")
    nearest = nearest_gt_distance_buckets(pred_np, gt_np, args.n_frames, _ORIGINAL_SCALE, args.max_distance)
    d = args.max_distance
    buckets = [
        (f"<= {d}um (Hungarian match candidates)", nearest <= d),
        (f"{d}-{2*d}um (near-duplicate zone)", (nearest > d) & (nearest <= 2 * d)),
        (f"{2*d}-{4*d}um", (nearest > 2 * d) & (nearest <= 4 * d)),
        (f"> {4*d}um (background zone)", nearest > 4 * d),
    ]
    for label, mask in buckets:
        print(f"  {label}: {mask.sum()} ({100*mask.mean():.1f}%)")

    zarr_arr = zarr.open_group(str(args.zarr), mode="r")["0"]
    print("\n  Raw image intensity by bucket (tests: real dim cell vs duplicate vs pure noise):")
    intensity_report("GT cells", [tuple(int(v) for v in row) for row in gt_np], zarr_arr)
    for label, mask in buckets:
        idx = np.where(mask)[0]
        coords = [tuple(int(v) for v in pred_np[i]) for i in idx]
        intensity_report(label, coords, zarr_arr)

    rng = np.random.default_rng(0)
    Z, Y, X = zarr_arr.shape[1:]
    rand_coords = [
        (int(rng.integers(0, args.n_frames)), int(rng.integers(0, Z)), int(rng.integers(0, Y)), int(rng.integers(0, X)))
        for _ in range(200)
    ]
    intensity_report("Random background voxels", rand_coords, zarr_arr)

    # --- Part 2: was n_total computed correctly? ---
    print("\n=== Part 2: n_total sensitivity (raw annotated GT count vs estimated_number_of_nodes) ===")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = evaluate(pred, gt, scale=_ORIGINAL_SCALE, max_distance=args.max_distance)
    recall = node_recall(pred, gt)
    print(f"  {er}")
    print(f"  node_recall={recall:.4f}")

    from geff import GeffMetadata
    meta = GeffMetadata.read(args.gt)
    estimated_total = (meta.extra or {}).get("estimated_number_of_nodes")

    n_total_wrong = gt.num_nodes()
    m_wrong = per_sample_metrics(er, n_total=n_total_wrong, node_recall=recall)
    print(f"\n  Using n_total = raw annotated GT count in window ({n_total_wrong}):")
    print(f"    total_node_ratio={m_wrong['total_node_ratio']:.3f}  adj_edge_jaccard={m_wrong['adj_edge_jaccard']:.4f}")

    if estimated_total is not None:
        n_total_correct = estimated_total * (args.n_frames / args.total_frames)
        m_correct = per_sample_metrics(er, n_total=n_total_correct, node_recall=recall)
        print(
            f"\n  Using n_total = estimated_number_of_nodes prorated to window "
            f"({estimated_total} * {args.n_frames}/{args.total_frames} = {n_total_correct:.1f}):"
        )
        print(
            f"    total_node_ratio={m_correct['total_node_ratio']:.3f}  "
            f"adj_edge_jaccard={m_correct['adj_edge_jaccard']:.4f}"
        )
    else:
        print("\n  No estimated_number_of_nodes in this GEFF's metadata.extra -- cannot compute the corrected n_total.")


if __name__ == "__main__":
    main()
