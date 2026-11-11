"""Per-stage GPU times of CGQ's production broad phase, from the nsys trace of `cgq_side.py time`.

    python bench/ipc_bvh/cgq_nsys.py --scene kimono data/nsys/cgq_kimono.sqlite

Every repeat of the timing loop is one NVTX range `cgq frame=F rep=R` holding one recompute() pipeline launch, a
CUDA graph whose kernels nsys records one by one (--cuda-graph-trace=node). Each kernel is assigned a stage by name:

- build: the bounds of points, edges and faces, the body cull (group boxes, group tree, group queries, the pass-A
  query compaction) and the LBVH builds (tree M, tree C, group tree), i.e. EFBroadPhase::triangle_build and
  edge_build;
- query: pass A (traversal of tree M, sub-pair expansion, PT and EE leaf filters), pass B and the static one-ring
  list (EFBroadPhase::pt_query and ee_query), split into its PT filter, EE filter and the traversal they share;
- ph: halfplane_query_kernel (no BVH);
- narrow: the export classification twins (export_assemble_*), which also evaluate the barrier derivatives of the
  active pairs, unlike the classification-only narrow phase of the upstream side;
- other: the counter resets of the recompute pipeline, the CCD export kernels (excluded from every stage).

A stage time is the span from the start of its first kernel to the end of its last one in the repeat (what the stage
costs inside the Newton graph, launch gaps included); `broad_phase` is the span from the first build kernel to the
last query or ph kernel. Kernel sums are reported next to the spans.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

STAGE_PATTERNS = (
    ("narrow", r"^export_assemble_"),
    ("other", r"^export_|^pipeline_"),
    ("ph", r"^halfplane_query"),
    ("pt_filter", r"^bvh_filter_raw_pt"),
    ("ee_filter", r"^bvh_filter_raw_ee"),
    ("query_traverse", r"^ef_pass_|^ef_raw_reset|^bvh_raw_reset|^ef_static"),
    ("build", r"^ef_point_bounds|^ef_edge_bounds|^ef_face_bounds|^ef_c_bounds|^ef_body_|^des_|^bvh_|^drs_"),
)
QUERY_STAGES = ("query_traverse", "pt_filter", "ee_filter")


def stage_of(name: str) -> str:
    for stage, pattern in STAGE_PATTERNS:
        if re.search(pattern, name):
            return stage
    raise ValueError(f"kernel {name} has no stage")


def span(kernels: list[tuple[int, int]]) -> float:
    return (max(end for _, end in kernels) - min(start for start, _ in kernels)) * 1e-6 if kernels else 0.0


def total(kernels: list[tuple[int, int]]) -> float:
    return sum(end - start for start, end in kernels) * 1e-6


def parse(sqlite_path: Path) -> dict:
    db = sqlite3.connect(sqlite_path)
    ranges = db.execute(
        "select text, start, end from NVTX_EVENTS where text like 'cgq frame=% rep=%' order by start"
    ).fetchall()
    kernels = db.execute(
        "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName = s.id "
        "order by k.start"
    ).fetchall()
    starts = np.array([k[0] for k in kernels])
    frames = defaultdict(lambda: defaultdict(list))
    kernel_names = defaultdict(list)
    for text, t_begin, t_end in ranges:
        frame = int(re.search(r"frame=(\d+)", text).group(1))
        stages = defaultdict(list)
        lo, hi = np.searchsorted(starts, t_begin), np.searchsorted(starts, t_end)
        for k_start, k_end, name in kernels[lo:hi]:
            stages[stage_of(name)].append((k_start, k_end))
            kernel_names[name].append((k_end - k_start) * 1e-6)
        query = [k for stage in QUERY_STAGES for k in stages[stage]]
        broad = stages["build"] + query + stages["ph"]
        rep = {
            "build": span(stages["build"]),
            "query": span(query),
            "query_traverse": span(stages["query_traverse"]),
            "pt_filter": total(stages["pt_filter"]),
            "ee_filter": total(stages["ee_filter"]),
            "ph": span(stages["ph"]),
            "narrow": total(stages["narrow"]),
            "broad_phase": span(broad),
            "broad_phase_kernel_sum": total(broad),
            "n_broad_phase_kernels": len(broad),
        }
        for key, value in rep.items():
            frames[frame][key].append(value)
    result = {}
    for frame, samples in sorted(frames.items()):
        result[frame] = {
            key: {"median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90))}
            for key, values in samples.items()
        }
        result[frame]["repeats"] = len(samples["build"])
    kernels_summary = {
        name: {"launches": len(values), "median_ms": float(np.median(values)), "sum_ms": float(np.sum(values))}
        for name, values in sorted(kernel_names.items(), key=lambda item: -sum(item[1]))
    }
    return {"frames": result, "kernels": kernels_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    result = parse(args.sqlite)
    memory = HERE / "data" / args.scene / "cgq_memory.json"
    if memory.is_file():
        result["solver_memory"] = json.loads(memory.read_text(encoding="utf-8"))
    out = HERE / "results" / f"{args.scene}_cgq.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for frame, stages in result["frames"].items():
        print(
            f"[{args.scene}] frame {frame}: "
            + " ".join(f"{key}={stages[key]['median_ms']:.3f}" for key in ("build", "query", "ph", "narrow", "broad_phase"))
        )
    print(f"[{args.scene}] wrote {out}")


if __name__ == "__main__":
    main()
