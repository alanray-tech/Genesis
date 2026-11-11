"""Where the time of the bvh.py side goes on one frame: GPU kernel time against launches and host synchronizations.

    python bench/ipc_bvh/upstream_nsys.py --scene kimono --schedule deferred data/nsys/upstream_kimono_deferred.sqlite

Reads the nsys trace of `upstream_side.py --nsys-frame`, whose timed calls are NVTX ranges named after their stage
(build, pt_query, ee_query, ph_query, narrow, counts_readback, total). Per stage and repeat: the wall time of the range,
the summed duration and the span of its GPU kernels, the number of kernels and of CUDA runtime calls (launches,
copies, synchronizations). The medians land in results/<scene>_upstream_<schedule>_nsys_breakdown.json, with the
kernels of the `total` ranges by summed time.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGES = ("build", "pt_query", "ee_query", "ph_query", "narrow", "counts_readback", "total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--schedule", default="deferred")
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    ranges = db.execute("select text, start, end from NVTX_EVENTS where end is not null order by start").fetchall()
    kernels = db.execute(
        "select k.start, k.end, s.value, k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on k.shortName = s.id order by k.start"
    ).fetchall()
    runtime = db.execute(
        "select r.start, r.end, s.value, r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds s on r.nameId = s.id order by r.start"
    ).fetchall()
    runtime_starts = np.array([r[0] for r in runtime])
    kernels_by_correlation = defaultdict(list)
    for k_start, k_end, name, corr in kernels:
        kernels_by_correlation[corr].append((k_start, k_end, name))

    samples = defaultdict(lambda: defaultdict(list))
    total_kernels = defaultdict(list)
    for text, t_begin, t_end in ranges:
        if text not in STAGES:
            continue
        lo, hi = np.searchsorted(runtime_starts, t_begin), np.searchsorted(runtime_starts, t_end)
        calls = runtime[lo:hi]
        # The kernels a range launched, found through the correlation of their launch calls
        stage_kernels = [k for _, _, _, corr in calls for k in kernels_by_correlation.get(corr, ())]
        kernel_sum = sum(end - start for start, end, _ in stage_kernels) * 1e-6
        kernel_span = (
            (max(end for _, end, _ in stage_kernels) - min(start for start, _, _ in stage_kernels)) * 1e-6
            if stage_kernels
            else 0.0
        )
        call_names = [name for _, _, name, _ in calls]
        values = {
            "wall_ms": (t_end - t_begin) * 1e-6,
            "kernel_sum_ms": kernel_sum,
            "kernel_span_ms": kernel_span,
            "n_kernels": len(stage_kernels),
            "n_runtime_calls": len(calls),
            "n_launches": sum("Launch" in name for name in call_names),
            "n_syncs_and_copies": sum(("Synchronize" in name) or ("Memcpy" in name) for name in call_names),
        }
        for key, value in values.items():
            samples[text][key].append(value)
        if text == "total":
            for start, end, name in stage_kernels:
                total_kernels[name].append((end - start) * 1e-6)
    breakdown = {
        stage: {key: float(np.median(values)) for key, values in samples[stage].items()} for stage in STAGES
    }
    n_totals = len(samples["total"]["wall_ms"])
    breakdown["total_kernels"] = {
        name: {"per_repeat_ms": float(np.sum(values)) / n_totals, "launches_per_repeat": len(values) / n_totals}
        for name, values in sorted(total_kernels.items(), key=lambda item: -sum(item[1]))
    }
    out = HERE / "results" / f"{args.scene}_upstream_{args.schedule}_nsys_breakdown.json"
    out.write_text(json.dumps(breakdown, indent=1), encoding="utf-8")
    for stage in STAGES:
        print(f"[{args.scene}/{args.schedule}] {stage}: " + json.dumps({k: round(v, 3) for k, v in breakdown[stage].items()}))
    for name, value in list(breakdown["total_kernels"].items())[:12]:
        print(f"  {value['per_repeat_ms']:.3f} ms x{value['launches_per_repeat']:.0f}  {name[:110]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
