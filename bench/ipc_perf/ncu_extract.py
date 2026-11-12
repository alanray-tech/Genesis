"""Extract the per-launch metrics of an Nsight Compute report (run_ncu.ps1) and summarize them per kernel.

For every kernel name: the launches, then for the median launch by duration and for the slowest one: duration, grid x
block, registers per thread, theoretical and achieved occupancy, local-memory traffic (spills), SM and memory throughput,
fp64 pipe utilization, threads active per instruction (divergence), uniform-branch share, global atomics, and the top
three stall reasons. Writes <report>.json next to the report.

    python ncu_extract.py results/ncu_genesis_crossed_drop_f70_main.ncu-rep [--min-us 5]
"""

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

import numpy as np

NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe"

parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--min-us", type=float, default=5.0, help="kernels whose median launch is shorter are listed apart")
args = parser.parse_args()

raw = subprocess.run([NCU, "-i", args.report, "--page", "raw", "--csv"], capture_output=True, text=True, check=True).stdout
reader = csv.reader(io.StringIO(raw))
header = next(reader)
units = next(reader)
rows = [dict(zip(header, row)) for row in reader]


def number(row: dict, key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return float("nan")


def unit_scale(key: str, to: str) -> float:
    unit = units[header.index(key)] if key in header else ""
    scales = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "nsecond": 1e-3, "ms": 1e3, "msecond": 1e3, "byte": 1.0}
    return scales.get(unit, 1.0) if to == "us" else 1.0


def first(row: dict, keys) -> float:
    for key in keys:
        if key in row and row[key] not in ("", "n/a"):
            return number(row, key)
    return float("nan")


stall_keys = [
    key
    for key in header
    if key.startswith("smsp__average_warps_issue_stalled_") and key.endswith("_per_issue_active.ratio")
]
fp64_keys = [key for key in header if "fp64" in key and "pct_of_peak_sustained_active" in key]


def launch_metrics(row: dict) -> dict:
    duration_key = "gpu__time_duration.sum"
    stalls = sorted(
        ((key.removeprefix("smsp__average_warps_issue_stalled_").removesuffix("_per_issue_active.ratio"), number(row, key))
         for key in stall_keys),
        key=lambda item: -(item[1] if item[1] == item[1] else 0.0),
    )
    return {
        "us": number(row, duration_key) * unit_scale(duration_key, "us"),
        "grid": row.get("launch__grid_size"),
        "block": row.get("launch__block_size"),
        "regs": number(row, "launch__registers_per_thread"),
        "occupancy_theoretical": number(row, "sm__maximum_warps_per_active_cycle_pct"),
        "occupancy_achieved": number(row, "sm__warps_active.avg.pct_of_peak_sustained_active"),
        "local_load_bytes": first(row, ["l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"]),
        "local_store_bytes": first(row, ["l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"]),
        "sm_throughput_pct": number(row, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "memory_throughput_pct": number(row, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
        "fp64_pipe_pct": max((number(row, key) for key in fp64_keys), default=float("nan")),
        "threads_per_inst": number(row, "smsp__thread_inst_executed_per_inst_executed.ratio"),
        "branch_uniform_pct": number(row, "smsp__sass_average_branch_targets_threads_uniform.pct"),
        "global_atomics": first(row, ["smsp__sass_inst_executed_op_global_atom.sum"])
        + first(row, ["smsp__sass_inst_executed_op_global_red.sum"]),
        "instructions": number(row, "smsp__inst_executed.sum"),
        "top_stalls": [f"{name} {value:.1f}" for name, value in stalls[:3]],
    }


by_kernel: dict[str, list[dict]] = {}
for row in rows:
    by_kernel.setdefault(row.get("Kernel Name", "?"), []).append(launch_metrics(row))

summary = {}
for name, launches in by_kernel.items():
    durations = np.array([launch["us"] for launch in launches])
    order = np.argsort(durations)
    summary[name] = {
        "launches": len(launches),
        "median": launches[order[len(order) // 2]],
        "slowest": launches[order[-1]],
        "fastest_us": float(durations.min()),
    }
Path(args.report).with_suffix(".json").write_text(json.dumps(summary, indent=1))

fields = (
    "us", "regs", "occupancy_theoretical", "occupancy_achieved", "local_load_bytes", "local_store_bytes",
    "sm_throughput_pct", "memory_throughput_pct", "fp64_pipe_pct", "threads_per_inst", "branch_uniform_pct",
    "global_atomics",
)
short = ("us", "regs", "occ_th", "occ_ach", "local_ld", "local_st", "sm%", "mem%", "fp64%", "thr/inst", "unif%", "atom")
print("kernel | which | launches | " + " | ".join(short) + " | grid x block | top stalls")
for name, entry in sorted(summary.items(), key=lambda item: -item[1]["median"]["us"] * item[1]["launches"]):
    if entry["median"]["us"] < args.min_us and entry["slowest"]["us"] < args.min_us:
        continue
    for which in ("median", "slowest"):
        launch = entry[which]
        values = " | ".join(f"{launch[key]:.4g}" for key in fields)
        print(
            f"{name[:60]} | {which} | {entry['launches']} | {values} | {launch['grid']} x {launch['block']} | "
            f"{'; '.join(launch['top_stalls'])}"
        )
