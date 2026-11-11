"""Merge the results of both sides into results/frames.csv, results/summary.json and results/tables.md.

    python bench/ipc_bvh/make_report.py

Per frame (frames.csv): candidate, raw-hit and active-pair counts of both sides, the active-set and candidate-set
differences, and the median and p90 of every stage. Per scene (summary.json, tables.md): the median over the frames
of each per-frame median and p90, the frame with the most CGQ candidates, and the ratio upstream / CGQ.

Stage definitions (see cgq_nsys.py and upstream_side.py):
- CGQ build, query, ph, broad phase: GPU spans of the production kernels inside the recompute() graph (nsys);
- CGQ narrow: CGQ's classification-only kernels (cgq_filter.cu) on CGQ's candidates, CUDA events;
- upstream stages: GPU-synchronized wall times of each Quadrants call sequence; total: the whole sequence with one
  synchronization, narrow phase and candidate-count readback included.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCENES = ("two_bunnies", "cloth_stack", "wrecking_ball", "kimono", "cloth_stack_x3")
SCHEDULES = ("deferred", "inline")
UPSTREAM_STAGES = ("build", "pt_query", "ee_query", "ph_query", "narrow", "counts_readback", "total")
CGQ_STAGES = ("build", "query", "query_traverse", "pt_filter", "ee_filter", "ph", "broad_phase", "narrow")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def scene_info(scene: str) -> dict:
    with np.load(HERE / "data" / scene / "frames.npz") as frames:
        return {
            "n_verts": int(frames["n_verts"]),
            "n_surf_verts": len(frames["surf_verts"]),
            "n_edges": len(frames["surf_edges"]),
            "n_faces": len(frames["surf_faces"]),
            "n_bodies": len(frames["self_collision"]),
            "n_halfplanes": len(frames["halfplane_normals"]),
            "d_hat": float(frames["d_hat"]),
            "cgq_revision": str(frames["cgq_revision"]),
            "bvh_config": json.loads(str(frames["bvh_config"])),
        }


def cgq_bytes_estimate(info: dict, n_pt: int, n_ee: int) -> dict:
    """Device bytes of CGQ's EF broad phase (ef_broad_phase.cu) at the demand of the largest frame, an estimate.

    Bounds: DOP14f boxes (64 B) of the points, edges, faces and face proxies, fp64 point and edge gaps. Tree M: per
    node a DOP14f box and a 16 B node record, per leaf a 64-bit morton key, an index and their sort copies. Pairs: the
    PT and EE pair buffers at the demand, the pass-A hit and sub-pair buffers at their initial factor (2x the pairs).
    """
    n_sv, n_e, n_f = info["n_surf_verts"], info["n_edges"], info["n_faces"]
    bounds = n_sv * (64 + 8) + n_e * (64 + 8) + n_f * 2 * 64
    tree_m = (2 * n_f - 1) * (64 + 16) + n_f * 2 * (8 + 4)
    pairs = (n_pt + n_ee) * 2 * 4
    hits = 2 * (n_pt + n_ee) * 2 * 4 + 2 * (n_pt + n_ee) * 2 * 4
    return {"bounds": bounds, "tree_m": tree_m, "pair_buffers": pairs, "hit_and_sub_pair_buffers": hits}


def summarize(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def main() -> None:
    rows, summary = [], {}
    for scene in SCENES:
        cgq = load(RESULTS / f"{scene}_cgq.json")
        narrow = load(RESULTS / f"{scene}_cgq_narrow_deferred.json")
        upstream = {k: load(RESULTS / f"{scene}_upstream_{k}.json") for k in SCHEDULES}
        if cgq is None or narrow is None or upstream["deferred"] is None:
            print(f"[{scene}] results missing, skipped")
            continue
        info = scene_info(scene)
        narrow_by_frame = {r["frame"]: r for r in narrow["frames"]}
        scene_rows = []
        for i_row, up_row in enumerate(upstream["deferred"]["frames"]):
            frame = up_row["frame"]
            check = up_row["check"]
            cgq_frame = cgq["frames"][str(frame)]
            nar = narrow_by_frame[frame]
            row = {"scene": scene, "frame": frame}
            for ch in ("pt", "ee", "ph"):
                c = check[ch]
                row[f"{ch}_candidates_cgq"] = c["candidates_cgq"]
                row[f"{ch}_candidates_upstream"] = c["candidates_upstream"]
                row[f"{ch}_active"] = c["active_cgq"]
                row[f"{ch}_active_only_upstream"] = c["active_only_upstream"]
                row[f"{ch}_active_only_cgq"] = c["active_only_cgq"]
                row[f"{ch}_candidates_only_upstream"] = c["upstream_candidates_not_in_cgq"]
                row[f"{ch}_candidates_only_cgq"] = c["cgq_candidates_missing_upstream"]
                row[f"{ch}_quadrants_flag_mismatch"] = c["quadrants_flag_mismatch"]
            row["pt_raw_hits_upstream"] = check["raw_hits"]["pt"]
            row["ee_raw_hits_upstream"] = check["raw_hits"]["ee"]
            row["cgq_code_on_upstream_candidates_active_diff"] = sum(sum(d) for d in nar["upstream_vs_cgq_active"].values())
            for stage in CGQ_STAGES:
                if stage == "narrow":
                    continue
                row[f"cgq_{stage}_median_ms"] = cgq_frame[stage]["median_ms"]
                row[f"cgq_{stage}_p90_ms"] = cgq_frame[stage]["p90_ms"]
            row["cgq_narrow_export_median_ms"] = cgq_frame["narrow"]["median_ms"]
            for key, label in (("narrow_on_cgq_candidates", "cgq_narrow"), ("narrow_on_upstream_candidates",
                                                                                "cgq_narrow_on_upstream_candidates")):
                row[f"{label}_median_ms"] = nar[key]["median_ms"]
                row[f"{label}_p90_ms"] = nar[key]["p90_ms"]
            row["cgq_total_median_ms"] = row["cgq_broad_phase_median_ms"] + row["cgq_narrow_median_ms"]
            row["cgq_total_p90_ms"] = row["cgq_broad_phase_p90_ms"] + row["cgq_narrow_p90_ms"]
            for schedule, result in upstream.items():
                if result is None:
                    continue
                timing = result["frames"][i_row]["timing"]
                assert result["frames"][i_row]["frame"] == frame
                for stage in UPSTREAM_STAGES:
                    row[f"up_{schedule}_{stage}_median_ms"] = timing[stage]["median_ms"]
                    row[f"up_{schedule}_{stage}_p90_ms"] = timing[stage]["p90_ms"]
                row[f"up_{schedule}_query_median_ms"] = timing["pt_query"]["median_ms"] + timing["ee_query"]["median_ms"]
                row[f"up_{schedule}_query_p90_ms"] = timing["pt_query"]["p90_ms"] + timing["ee_query"]["p90_ms"]
            scene_rows.append(row)
        rows += scene_rows

        def med(key):
            return summarize([r[key] for r in scene_rows if key in r])

        largest = max(scene_rows, key=lambda r: r["pt_candidates_cgq"] + r["ee_candidates_cgq"])
        n_pt_max = max(r["pt_candidates_cgq"] for r in scene_rows)
        n_ee_max = max(r["ee_candidates_cgq"] for r in scene_rows)
        scene_summary = {
            "info": info,
            "frames": len(scene_rows),
            "largest_frame": largest["frame"],
            "pass_active_sets_identical": all(
                r[f"{ch}_active_only_upstream"] == 0 and r[f"{ch}_active_only_cgq"] == 0
                for r in scene_rows
                for ch in ("pt", "ee", "ph")
            ),
            "pass_cgq_code_on_upstream_candidates": narrow["upstream_active_equals_cgq"],
            "extension_reproduces_recompute": narrow["extension_reproduces_recompute"],
            "cgq_candidates_all_found_upstream": all(
                r[f"{ch}_candidates_only_cgq"] == 0 for r in scene_rows for ch in ("pt", "ee", "ph")
            ),
            "candidates_only_upstream_total": sum(
                r[f"{ch}_candidates_only_upstream"] for r in scene_rows for ch in ("pt", "ee", "ph")
            ),
            "candidates_cgq_total": sum(r[f"{ch}_candidates_cgq"] for r in scene_rows for ch in ("pt", "ee", "ph")),
            "active_total": sum(r[f"{ch}_active"] for r in scene_rows for ch in ("pt", "ee", "ph")),
            "median_over_frames": {key: med(key) for key in scene_rows[0] if key.endswith("_ms")},
            "largest_frame_row": largest,
            "memory_bytes": {
                "upstream": {k: upstream[k]["memory_bytes"] for k in SCHEDULES if upstream[k]},
                "cgq_estimate": cgq_bytes_estimate(info, n_pt_max, n_ee_max),
                "cgq_solver_total": cgq.get("solver_memory"),
            },
            "cgq_recompute_wall_ms": {
                label: summarize(list((load(HERE / "data" / scene / f"cgq_recompute_wall_{label}.json") or {}).values()))
                for label in ("traced", "untraced")
            },
            "upstream_nsys": {k: load(RESULTS / f"{scene}_upstream_{k}_nsys_breakdown.json") for k in SCHEDULES},
        }
        summary[scene] = scene_summary
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(RESULTS / "frames.csv", "w", newline="", encoding="utf-8") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "tables.md").write_text(tables(summary), encoding="utf-8")
    print((RESULTS / "tables.md").read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.3f}" if value < 10 else f"{value:.1f}"


def tables(summary: dict) -> str:
    lines = ["## Scenes", "", "| scene | verts | surf verts | edges | faces | bodies | frames | CGQ candidates / frame | active / frame |",
             "|---|---|---|---|---|---|---|---|---|"]
    for scene, s in summary.items():
        i = s["info"]
        lines.append(
            f"| {scene} | {i['n_verts']} | {i['n_surf_verts']} | {i['n_edges']} | {i['n_faces']} | {i['n_bodies']} | "
            f"{s['frames']} | {s['candidates_cgq_total'] / s['frames']:.0f} | {s['active_total'] / s['frames']:.0f} |"
        )
    lines += ["", "## Correctness", "",
              "| scene | active sets identical (port) | identical (CGQ kernels on bvh.py candidates) | "
              "CGQ candidates all found | candidates only in bvh.py (all frames) |", "|---|---|---|---|---|"]
    for scene, s in summary.items():
        lines.append(
            f"| {scene} | {s['pass_active_sets_identical']} | {s['pass_cgq_code_on_upstream_candidates']} | "
            f"{s['cgq_candidates_all_found_upstream']} | {s['candidates_only_upstream_total']} |"
        )
    for schedule in SCHEDULES:
        lines += ["", f"## Timing, median over frames of the per-frame median (p90), ms, bvh.py {schedule}", "",
                  "| scene | stage | CGQ | bvh.py | bvh.py / CGQ |", "|---|---|---|---|---|"]
        pairs = (
            ("build", "cgq_build", f"up_{schedule}_build"),
            ("PT + EE query", "cgq_query", f"up_{schedule}_query"),
            ("  PT query", None, f"up_{schedule}_pt_query"),
            ("  EE query", None, f"up_{schedule}_ee_query"),
            ("PH query", "cgq_ph", f"up_{schedule}_ph_query"),
            ("narrow", "cgq_narrow", f"up_{schedule}_narrow"),
            ("count readback", None, f"up_{schedule}_counts_readback"),
            ("total", "cgq_total", f"up_{schedule}_total"),
        )
        for scene, s in summary.items():
            m = s["median_over_frames"]
            if f"up_{schedule}_total_median_ms" not in m:
                continue
            for label, cgq_key, up_key in pairs:
                up_med, up_p90 = m[f"{up_key}_median_ms"], m[f"{up_key}_p90_ms"]
                if cgq_key is None:
                    lines.append(f"| {scene} | {label} | | {fmt(up_med)} ({fmt(up_p90)}) | |")
                    continue
                c_med, c_p90 = m[f"{cgq_key}_median_ms"], m[f"{cgq_key}_p90_ms"]
                ratio = f"{up_med / c_med:.1f}x" if c_med > 0 else ""
                lines.append(f"| {scene} | {label} | {fmt(c_med)} ({fmt(c_p90)}) | {fmt(up_med)} ({fmt(up_p90)}) | {ratio} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
