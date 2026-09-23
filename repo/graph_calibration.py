"""Stage 4: graph calibration -- motion relink, gap closing, safe divisions,
short-track pruning, trajectory smoothing.

Direct port of the "graph calibration" cell (cell 11, ~1480 lines) from the
reference notebook `biohub-exp073-gap-5-8-public.ipynb`, documented in full
in README.md section 4. Per that README (section 1): "The 0.902 result comes
almost entirely from stage 4 -- a deterministic, hand-tuned graph
post-processing/calibration layer applied to the raw ILP output." This module
is that layer, decoupled from the Kaggle-notebook env-var config plumbing
(``CalibrationConfig`` below holds the same values as plain fields) and from
pilkwang's own stage-1-3 model -- it operates purely on the plain
``nodes_by_id`` / ``raw_edges`` structures described in README section 4.2,
so it works unchanged on top of *our* stage-1-3 output (repo/predict.py).

Ported as-is (same algorithm, same default numeric constants -- the
"Selected (production) calibration" values from README section 3.1, falling
back to README section 3.2 defaults for anything not explicitly overridden
in the scored 0.902 run). Two functional exceptions, both disclosed:

1. The DeepCenter veto gate (README section 6) is omitted entirely -- it
   requires a second, separate trained model, was disabled in the scored run
   (`BIOHUB_USE_DEEPCENTER_VETO=0`), and README's own diagnostics prove it
   checked zero nodes/edges even when enabled. Every ``deepcenter_*`` call
   site in the reference becomes a no-op here, matching the disabled
   behavior exactly.
2. ``read_frame`` uses this repo's existing zarr-reading convention
   (``zarr.open_group(path)["0"][t]``, as in ``datasets.py``) instead of the
   reference's manual blosc2 chunk-decompression fast path, which assumes a
   specific Zarr v3 sharding layout. Same data, more robust read path.

Known caveat (see SUBMISSION_PIPELINE.md): every distance/radius/cap constant
below was tuned against *pilkwang's* detector's output distribution, not
ours. Ported verbatim as the correct starting point, not re-validated here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment

VOXEL_SCALE_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)  # (z, y, x)

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


@dataclass
class CalibrationConfig:
    """Stage-4 knobs. Defaults are the README section 3.1 "selected (production)
    calibration" values, falling back to section 3.2 defaults where not
    explicitly overridden in the scored 0.902 run."""

    # Basic edge filtering
    output_edge_max_um: float = 14.0
    output_enforce_next_frame: bool = True
    output_single_parent_repair: bool = True
    output_single_child_repair: bool = False  # off: divisions rely on this staying off
    output_prune_isolated: bool = True

    # Motion-based relinking
    output_motion_relink: bool = True
    # 4.0/8.0 um, tighter than pilkwang's 6.0/10.0 -- STAGE4_SWEEP.md found this
    # improves adj_edge_jaccard (0.9961 -> 0.9981) on our own pipeline's output
    # with identical node_recall/edge_jaccard (1.000/1.000), i.e. it's not an
    # artifact of the total_node_ratio formula's sign flip (see that doc's
    # "metric-gaming" section) -- it's a real reduction in spurious long-range
    # motion-relink matches, consistent with our ILP already producing very
    # clean short-range associations that don't need a wide relink net.
    motion_relink_tight_um: float = 4.0
    motion_relink_relaxed_um: float = 8.0
    motion_relink_velocity_weight: float = 0.5
    motion_relink_learned_bonus: float = 1.0  # production override (default 0.75)
    motion_relink_max_frame_nodes: int = 2600

    # Division geometry filter (post-hoc safety net; redundant given safe-division gating)
    output_division_geometry_filter: bool = False
    div_parent_max_um: float = 10.5
    div_sister_max_um: float = 8.0
    div_drop_to_single_if_bad: bool = True

    # Single-frame gap closing
    output_gap_close: bool = True
    gap_close_max_gap: int = 1  # requested 2 in production, code clamps effective to 1
    gap_close_um: float = 5.8  # production override (default 6.0)
    gap_close_reuse_existing: bool = True
    gap_close_reuse_um: float = 3.2
    gap_close_max_added_frac: float = 0.05
    gap_close_max_added_abs: int = 2000
    gap_refine_synthetic: bool = True
    gap_refine_win_z: int = 1
    gap_refine_win_yx: int = 3
    gap_refine_max_shift_um: float = 3.2

    # Short-track filtering
    output_filter_short_tracks: bool = True
    output_min_track_len: int = 6
    output_keep_division_components: bool = True
    adaptive_short_track_rescue: bool = False
    short_track_rescue_trigger_removed_frac: float = 0.10
    short_track_rescue_min_len: int = 4
    short_track_rescue_min_mean_edge_prob: float = 0.82
    short_track_rescue_max_mean_edge_dist_um: float = 3.25
    short_track_rescue_max_nodes_frac: float = 0.018
    short_track_rescue_max_nodes_abs: int = 180

    # Trajectory smoothing
    output_linefit_smooth: bool = True
    output_linefit_weight: float = 0.8
    output_linefit_window: int = 2

    # Strict 2-frame gap recovery (disabled in the scored run; ported for completeness)
    output_gap2_recovery: bool = False
    gap2_max_total_um: float = 10.2
    gap2_max_step_um: float = 4.4
    gap2_max_links_frac: float = 0.0045
    gap2_max_links_abs: int = 180
    gap2_require_context: bool = True
    gap2_frame_frac_cap: float = 0.006

    # Safe (conservative) division insertion
    output_safe_divisions: bool = True
    safe_div_max_um: float = 4.66  # production override (default 4.7)
    safe_div_sister_max_um: float = 8.5  # production override (default 7.2)
    safe_div_existing_child_max_um: float = 7.65  # production override (default 7.8)
    safe_div_frame_frac_cap: float = 0.0076  # production override (default 0.008)
    safe_div_global_frac_cap: float = 0.00375  # production override (default 0.004)


def _stats_defaultdict() -> dict:
    from collections import defaultdict
    return defaultdict(int)


# =============================================================================
# Geometry helpers
# =============================================================================

def edge_distance_um(source: dict, target: dict) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1


def _position_um(node: dict) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def read_frame(zarr_path: Path, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    """Read one raw frame (Z, Y, X) from a .zarr video, with per-call caching."""
    if t in frame_cache:
        return frame_cache[t]
    frame = np.asarray(zarr.open_group(str(zarr_path), mode="r")["0"][t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_midpoint(
    zarr_path: Path | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict,
    cfg: CalibrationConfig,
) -> tuple[float, float, float]:
    """Background-subtracted intensity-weighted centroid refinement of a synthetic gap midpoint."""
    if not cfg.gap_refine_synthetic or zarr_path is None:
        return midpoint
    try:
        frame = read_frame(zarr_path, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - cfg.gap_refine_win_z)
        z1 = min(frame.shape[0], z + cfg.gap_refine_win_z + 1)
        y0 = max(0, y - cfg.gap_refine_win_yx)
        y1 = min(frame.shape[1], y + cfg.gap_refine_win_yx + 1)
        x0 = max(0, x - cfg.gap_refine_win_yx)
        x1 = min(frame.shape[2], x + cfg.gap_refine_win_yx + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > cfg.gap_refine_max_shift_um:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint


# =============================================================================
# Motion-based relinking
# =============================================================================

def motion_relink_edges(
    nodes_by_id: dict[int, dict],
    stats: dict,
    cfg: CalibrationConfig,
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict]:
    """Discards the raw edges and rebuilds every frame-to-frame association via
    a two-pass (tight then relaxed) Hungarian assignment on
    ``motion_distance + 0.05*raw_distance - learned_bonus*edge_prob``."""
    if not cfg.output_motion_relink or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > cfg.motion_relink_max_frame_nodes:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict] = []

    def assign_pass(source_ids: list[int], target_ids: list[int], gate_um: float):
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            predicted = source_pos if prev_pos is None else source_pos + cfg.motion_relink_velocity_weight * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - cfg.motion_relink_learned_bonus * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((source_ids[int(r)], target_ids[int(c)], float(raw_dist[r, c]), float(motion_dist[r, c]), float(prob_matrix[r, c])))
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches = []
        for pass_name, gate_um in (("tight", cfg.motion_relink_tight_um), ("relaxed", cfg.motion_relink_relaxed_um)):
            pass_sources = [nid for nid in source_ids if nid in unmatched_sources]
            pass_targets = [nid for nid in target_ids if nid in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                stats["motion_relink_tight_edges" if pass_name == "tight" else "motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id, "target_id": target_id, "edge_prob": prob,
                "distance_um": raw, "motion_distance_um": motion,
                "motion_relinked": 1, "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges


# =============================================================================
# Single-frame gap closing
# =============================================================================

def close_single_frame_gaps(
    nodes_by_id: dict[int, dict],
    edges: list[dict],
    stats: dict,
    cfg: CalibrationConfig,
    zarr_path: Path | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
) -> tuple[dict[int, dict], list[dict]]:
    if not cfg.output_gap_close or cfg.gap_close_max_gap < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(e["source_id"]) for e in edges}
    incoming = {int(e["target_id"]) for e in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        cfg.gap_close_max_added_abs,
        max(1, int(round(len(nodes_by_id) * cfg.gap_close_max_added_frac))) if cfg.gap_close_max_added_frac > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict] = []

    effective_gap_max = min(cfg.gap_close_max_gap, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = cfg.gap_close_um * (gap + 1)
            d = np.zeros((len(end_ids), len(start_ids)), dtype=np.float64)
            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)
            stats["gap_candidates"] += int((d <= threshold_um).sum())
            if not np.isfinite(d).any():
                continue

            big = threshold_um * 1000.0 + 1.0
            cost = np.where(d <= threshold_um, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if d[r, c] > threshold_um:
                    continue
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id = None
                middle_reused = False
                if cfg.gap_close_reuse_existing:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= cfg.gap_close_reuse_um:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(zarr_path, mid_t, mid_point, frame_cache, stats, cfg)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id, "t": mid_t,
                        "z": refined_point[0], "y": refined_point[1], "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                new_edges.append({
                    "source_id": source_id, "target_id": middle_id, "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle), "gap_closed": 1,
                })
                new_edges.append({
                    "source_id": middle_id, "target_id": target_id, "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target), "gap_closed": 1,
                })
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for e in edges:
        by_source.setdefault(int(e["source_id"]), []).append(int(e["target_id"]))
    return {s: t[0] for s, t in by_source.items() if len(t) == 1}


def _single_predecessor_map(edges: list[dict]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for e in edges:
        by_target.setdefault(int(e["target_id"]), []).append(int(e["source_id"]))
    return {t: s[0] for t, s in by_target.items() if len(s) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict], edges: list[dict], stats: dict, cfg: CalibrationConfig,
    zarr_path: Path | None = None,
) -> tuple[dict[int, dict], list[dict]]:
    """Disabled by default (matches the scored 0.902 run); ported for completeness."""
    if not cfg.output_gap2_recovery or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(e["source_id"]) for e in edges}
    incoming = {int(e["target_id"]) for e in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(cfg.gap2_max_links_abs, max(1, int(round(len(edges) * cfg.gap2_max_links_frac))))
    proposals = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > cfg.gap2_max_total_um or dist / 3.0 > cfg.gap2_max_step_um:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if cfg.gap2_require_context:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * cfg.gap2_frame_frac_cap)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(zarr_path, mid_t, midpoint, frame_cache, stats, cfg)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {"node_id": node_id, "t": mid_t, "z": refined_point[0], "y": refined_point[1], "x": refined_point[2]}
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id, "target_id": node_id, "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current), "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id, "target_id": start_id, "edge_prob": None,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target), "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


# =============================================================================
# Conservative division insertion
# =============================================================================

def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict], edges: list[dict], stats: dict, cfg: CalibrationConfig,
) -> list[dict]:
    """Add-only: proposes a second child for single-child parents whose existing
    link is already tight, never removes or alters the first child."""
    if not cfg.output_safe_divisions or not edges or not nodes_by_id:
        return edges

    out_by_source: dict[int, list[dict]] = {}
    incoming: set[int] = set()
    for e in edges:
        out_by_source.setdefault(int(e["source_id"]), []).append(e)
        incoming.add(int(e["target_id"]))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * cfg.safe_div_global_frac_cap)))
    added: list[dict] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [nid for nid in ids_by_t[t] if len(out_by_source.get(nid, [])) == 1]
        candidate_ids = [nid for nid in child_frame_ids if nid not in incoming and nid not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        frame_cap = max(1, int(round(len(source_ids) * cfg.safe_div_frame_frac_cap)))
        proposals = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > cfg.safe_div_existing_child_max_um:
                continue
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > cfg.safe_div_max_um:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > cfg.safe_div_sister_max_um:
                    continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            added.append({"source_id": source_id, "target_id": candidate_id, "edge_prob": None, "distance_um": parent_dist, "safe_division": 1})
            used_targets.add(candidate_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


# =============================================================================
# Short-track filtering
# =============================================================================

def filter_short_track_components(
    nodes_by_id: dict[int, dict], edges: list[dict], stats: dict, cfg: CalibrationConfig,
) -> tuple[dict[int, dict], list[dict]]:
    if not cfg.output_filter_short_tracks or cfg.output_min_track_len <= 1 or not edges:
        return nodes_by_id, edges

    parent = {nid: nid for nid in nodes_by_id}

    def find(nid: int) -> int:
        while parent[nid] != nid:
            parent[nid] = parent[parent[nid]]
            nid = parent[nid]
        return nid

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for e in edges:
        source_id, target_id = int(e["source_id"]), int(e["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for nid in nodes_by_id:
        components.setdefault(find(nid), []).append(nid)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(nid, 0) >= 2 for nid in members)
        if len(members) >= cfg.output_min_track_len or (cfg.output_keep_division_components and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if cfg.adaptive_short_track_rescue:
        component_edges: dict[int, list[dict]] = {root: [] for root in components}
        for e in edges:
            sid, tid = int(e["source_id"]), int(e["target_id"])
            if sid in parent and tid in parent:
                component_edges.setdefault(find(sid), []).append(e)

        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= cfg.short_track_rescue_trigger_removed_frac:
            budget = min(cfg.short_track_rescue_max_nodes_abs, max(0, int(round(len(nodes_by_id) * cfg.short_track_rescue_max_nodes_frac))))
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < cfg.short_track_rescue_min_len or len(members) >= cfg.output_min_track_len:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs, dists = [], []
                for e in c_edges:
                    try:
                        prob = float(e.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(e.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < cfg.short_track_rescue_min_mean_edge_prob or mean_dist > cfg.short_track_rescue_max_mean_edge_dist_um:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {nid: n for nid, n in nodes_by_id.items() if nid in keep}
    kept_edges = [e for e in edges if int(e["source_id"]) in kept_nodes and int(e["target_id"]) in kept_nodes]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


# =============================================================================
# Trajectory smoothing
# =============================================================================

def linefit_smooth_output_graph(nodes_by_id: dict[int, dict], edges: list[dict], stats: dict, cfg: CalibrationConfig) -> dict[int, dict]:
    """Topology-preserving: blends each node's position 80% toward a locally-linear
    fit through up to 5 unambiguous single-predecessor/successor neighbors."""
    if not cfg.output_linefit_smooth or cfg.output_linefit_weight <= 0 or cfg.output_linefit_window <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for e in edges:
        source_id, target_id = int(e["source_id"]), int(e["target_id"])
        source, target = nodes_by_id.get(source_id), nodes_by_id.get(target_id)
        if source is None or target is None or int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {nid: np.array([float(n["z"]), float(n["y"]), float(n["x"])], dtype=np.float64) for nid, n in nodes_by_id.items()}
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(cfg.output_linefit_weight, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, cfg.output_linefit_window + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, cfg.output_linefit_window + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


# =============================================================================
# Orchestrator
# =============================================================================

def calibrate_graph(
    nodes_by_id: dict[int, dict],
    raw_edges: list[dict],
    cfg: CalibrationConfig | None = None,
    zarr_path: Path | None = None,
) -> tuple[dict[int, dict], list[dict], dict]:
    """Full stage-4 pipeline, same call order as the reference's `filter_output_graph`:

    basic edge filter -> motion relink -> single-parent repair -> [single-child
    repair, off] -> gap close -> gap2 recovery [off] -> safe divisions ->
    [division geometry filter, off] -> prune isolated -> short-track filter ->
    linefit smoothing.

    Parameters
    ----------
    nodes_by_id : {node_id: {"node_id","t","z","y","x"}}
    raw_edges : [{"source_id","target_id","edge_prob"}, ...]
    zarr_path : path to the video's .zarr, for pixel-level gap-midpoint refinement
        (optional -- gap closing still works without it, just without refinement).
    """
    cfg = cfg or CalibrationConfig()
    stats = _stats_defaultdict()
    stats["raw_edges"] = len(raw_edges)

    edges: list[dict] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if cfg.output_enforce_next_frame and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge = dict(edge)
        edge["distance_um"] = distance_um
        if cfg.output_edge_max_um > 0 and distance_um > cfg.output_edge_max_um:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if cfg.output_motion_relink:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, cfg, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if cfg.output_single_parent_repair and edges:
        best_by_target: dict[int, dict] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(e) for e in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for e in edges if id(e) not in kept_ids)
        edges = [e for e in edges if id(e) in kept_ids]

    if cfg.output_single_child_repair and edges:
        best_by_source: dict[int, dict] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(e) for e in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for e in edges if id(e) not in kept_ids)
        edges = [e for e in edges if id(e) in kept_ids]

    frame_cache: dict[int, np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(nodes_by_id, edges, stats, cfg, zarr_path=zarr_path, frame_cache=frame_cache)
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, cfg, zarr_path=zarr_path)
    edges = add_safe_divisions_postlink(nodes_by_id, edges, stats, cfg)

    if cfg.output_division_geometry_filter and edges:
        by_source: dict[int, list[dict]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)
        filtered = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue
            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            top1, top2 = ranked[0], ranked[1]
            d1, d2 = float(top1["distance_um"]), float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            source = nodes_by_id[source_id]
            valid_division = (
                max(d1, d2) <= cfg.div_parent_max_um and sister <= cfg.div_sister_max_um
                and int(nodes_by_id[int(top1["target_id"])]["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])]["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif cfg.div_drop_to_single_if_bad:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if cfg.output_prune_isolated:
        incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
        if incident:
            kept_nodes = {nid: n for nid, n in nodes_by_id.items() if nid in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [e for e in edges if int(e["source_id"]) in nodes_by_id and int(e["target_id"]) in nodes_by_id]

    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats, cfg)
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats, cfg)

    return nodes_by_id, edges, dict(stats)


def calibrated_graph_to_rows(dataset: str, nodes_by_id: dict[int, dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert a calibrated (nodes_by_id, edges) pair into submission.csv row dicts
    (without the `id` index column -- callers assign that when writing)."""
    node_rows = []
    for node_id in sorted(nodes_by_id):
        node = nodes_by_id[node_id]
        node_rows.append({
            "dataset": dataset, "row_type": "node",
            "node_id": int(node["node_id"]), "t": int(node["t"]),
            "z": max(0, int(round(float(node["z"])))),
            "y": max(0, int(round(float(node["y"])))),
            "x": max(0, int(round(float(node["x"])))),
            "source_id": -1, "target_id": -1,
        })
    edge_rows = []
    for edge in edges:
        source_id, target_id = int(edge["source_id"]), int(edge["target_id"])
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise AssertionError(f"{dataset}: dangling edge after filtering")
        edge_rows.append({
            "dataset": dataset, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": source_id, "target_id": target_id,
        })
    return node_rows, edge_rows
