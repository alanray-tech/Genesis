"""Summarize an nsys report of genesis_profile.py / cgq_profile.py per step, from its SQLite export.

Per step (one NVTX range 'step k'): the wall time of the range, the GPU busy time (union of the kernel and memcpy
intervals), the kernel time summed, the kernel launches (kernels run, graph nodes included), the CUDA API calls by name
(launches, graph launches, synchronizations, copies), and the top kernels by time over the window with their launch
configuration.

    python nsys_summary.py results/genesis_crossed_drop.sqlite [--top 25]
"""

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("sqlite")
parser.add_argument("--top", type=int, default=25)
args = parser.parse_args()
db = sqlite3.connect(args.sqlite)
strings = dict(db.execute("SELECT id, value FROM StringIds"))
tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

steps = [
    (start, end, text)
    for start, end, text in db.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'step %' ORDER BY start")
]
marks = np.array([row[0] for row in db.execute("SELECT start FROM NVTX_EVENTS WHERE text = 'newton_iteration'")])
kernels = np.array(
    list(
        db.execute(
            "SELECT start, end, shortName, gridX, gridY, gridZ, blockX, blockY, blockZ, graphNodeId "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
    ),
    dtype=object,
)
memcpy = (
    np.array(list(db.execute("SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")), dtype=object)
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
    else np.zeros((0, 3), dtype=object)
)
api = list(db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"))


def busy(intervals) -> float:
    """The length of the union of the intervals, in ns."""
    total, current_end = 0, -1
    for start, end in sorted(intervals):
        if start > current_end:
            total += end - start
            current_end = end
        elif end > current_end:
            total += end - current_end
            current_end = end
    return total


per_step = []
api_totals = collections.Counter()
api_time = collections.Counter()
for start, end, text in steps:
    in_step = [k for k in kernels if start <= k[0] <= end]
    in_copy = [m for m in memcpy if start <= m[0] <= end]
    calls = collections.Counter()
    for a_start, a_end, name_id in api:
        if start <= a_start <= end:
            name = re.sub(r"_v\d+$", "", strings[name_id])
            calls[name] += 1
            api_time[name] += a_end - a_start
    api_totals.update(calls)
    n_marks = int(((marks >= start) & (marks <= end)).sum()) if len(marks) else 0
    per_step.append(
        {
            "step": text,
            "wall_ms": (end - start) / 1e6,
            "gpu_busy_ms": busy([(k[0], k[1]) for k in in_step] + [(m[0], m[1]) for m in in_copy]) / 1e6,
            "kernel_ms": sum(k[1] - k[0] for k in in_step) / 1e6,
            "kernels": len(in_step),
            "graph_nodes": sum(k[9] is not None and k[9] != 0 for k in in_step),
            "memcpy": len(in_copy),
            "newton_iterations_marked": n_marks,
            "api": dict(calls),
        }
    )

n_steps = len(per_step)
summary = {key: float(np.mean([s[key] for s in per_step])) for key in ("wall_ms", "gpu_busy_ms", "kernel_ms", "kernels", "graph_nodes", "memcpy")}
summary["wall_ms_median"] = float(np.median([s["wall_ms"] for s in per_step]))
summary["steps"] = n_steps
summary["newton_iterations_marked_per_step"] = float(np.mean([s["newton_iterations_marked"] for s in per_step]))
summary["api_calls_per_step"] = {name: count / n_steps for name, count in api_totals.most_common()}
summary["api_ms_per_step"] = {name: api_time[name] / n_steps / 1e6 for name, _ in api_totals.most_common()}

window = (steps[0][0], steps[-1][1]) if steps else (0, 0)
by_kernel = collections.defaultdict(lambda: [0, 0, set()])
for k in kernels:
    if window[0] <= k[0] <= window[1]:
        name = strings.get(k[2], str(k[2]))
        entry = by_kernel[name]
        entry[0] += k[1] - k[0]
        entry[1] += 1
        entry[2].add((k[3] * k[4] * k[5], k[6] * k[7] * k[8]))
top = sorted(by_kernel.items(), key=lambda item: -item[1][0])[: args.top]
summary["top_kernels"] = [
    {
        "name": name,
        "ms_per_step": total / n_steps / 1e6,
        "launches_per_step": count / n_steps,
        "avg_us": total / count / 1e3,
        "grid_x_block": sorted(configs)[:4],
    }
    for name, (total, count, configs) in top
]
out = Path(args.sqlite).with_suffix(".summary.json")
out.write_text(json.dumps({"summary": summary, "per_step": per_step}, indent=1))
print(json.dumps(summary, indent=1))
