"""Generate the coupling performance report assets.

The source measurements are transcribed from:

- dev-docs/contact-performance-debt.md
- Quadrants issue #945

Run from the repository root:

    python dev-docs/performance-report/generate_performance_report.py
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "coupling-performance-report.md"
DATA_PATH = REPORT_DIR / "coupling-performance-data.json"
PLOT_PATH = REPORT_DIR / "coupling-performance-summary.png"
BUNDLE_PATH = REPORT_DIR / "coupling-performance-report-2026-09-30.zip"


def metric(
    name: str,
    subsystem: str,
    before: float | None,
    after: float | None,
    unit: str,
    evidence: str,
    commit: str,
    scope: str,
    *,
    reported_speedup: float | None = None,
    caveat: str = "",
) -> dict[str, object]:
    speedup = reported_speedup
    reduction_percent = None
    if before is not None and after is not None:
        if after > 0.0:
            speedup = before / after
        if before > 0.0:
            reduction_percent = 100.0 * (before - after) / before
    elif speedup is not None:
        reduction_percent = 100.0 * (1.0 - 1.0 / speedup)
    return {
        "name": name,
        "subsystem": subsystem,
        "before": before,
        "after": after,
        "unit": unit,
        "speedup": speedup,
        "reduction_percent": reduction_percent,
        "evidence": evidence,
        "commit": commit,
        "scope": scope,
        "caveat": caveat,
    }


STAGE_METRICS = [
    metric(
        "Direct EE rank-1 scatter",
        "Contact assembly",
        149.0,
        84.0,
        "us",
        "pinned_cgq",
        "28b42c8f",
        "Frame-2 EE filter",
        caveat="Pinned-CGQ migration evidence; isolated kernel improved by up to 2.8x.",
    ),
    metric(
        "Warp-batched EE query/output",
        "Broad phase",
        403.8,
        60.1,
        "us",
        "pinned_cgq",
        "28b42c8f",
        "Orbit-mean complete warp-per-EE query schedule",
    ),
    metric(
        "Load-balanced dual EE",
        "Broad phase",
        None,
        None,
        "ratio",
        "pinned_cgq",
        "28b42c8f",
        "Kimono build plus query",
        reported_speedup=1.756,
        caveat="Production-auto query was 1.678x on Kimono and 1.008x on WreckingBall.",
    ),
    metric(
        "Warp-cooperative PT",
        "Broad phase",
        151.721,
        97.109,
        "us/Newton",
        "strict",
        "bcda51f4",
        "Frames 20-28, 18 Newton evaluations",
    ),
    metric(
        "Native-f32 scene bounds",
        "BVH",
        48.816,
        20.448,
        "us/Newton",
        "strict",
        "527ddf9a",
        "Frames 20-28, same retained f64 oracle",
    ),
    metric(
        "Direct internal refit",
        "BVH",
        199.870,
        132.655,
        "us/Newton",
        "strict",
        "527ddf9a",
        "f64 bounds held fixed",
    ),
    metric(
        "Packed DOP14f refit",
        "BVH",
        132.655,
        95.662,
        "us/Newton",
        "strict",
        "527ddf9a",
        "Direct refit before/after packing",
    ),
    metric(
        "Packed/direct broad phase",
        "BVH",
        608.777,
        422.971,
        "us/Newton",
        "strict",
        "527ddf9a",
        "Named broad-phase work",
    ),
    metric(
        "LBVH Morton OneSweep",
        "Sort",
        16.105,
        6.888,
        "ms/9 frames",
        "strict",
        "93be3590",
        "Two Morton sort stages, 18 Newton evaluations",
    ),
    metric(
        "OneSweep + scan + warp FSR",
        "Sort/reduce",
        24.881,
        12.534,
        "ms/9 frames",
        "strict",
        "ba777cda",
        "Doublet, triplet, and body sort/reduce",
    ),
    metric(
        "Fused rigid forest P/P^T",
        "Reduced KKT",
        134.0,
        35.0,
        "us/PCG",
        "strict",
        "910f9f4c",
        "Normalized expand/project cost",
    ),
    metric(
        "Fused proxy map/post",
        "Reduced KKT",
        34.4,
        14.5,
        "us/PCG",
        "strict",
        "910f9f4c",
        "Fixed-work proxy expansion/restriction/writeback",
    ),
    metric(
        "Remove duplicate native collision",
        "Rigid/contact routing",
        9.755,
        0.0073,
        "ms/9 frames",
        "strict",
        "e20e9f9d",
        "Native preprocessing task",
        caveat="End-to-end wall delta was small because the PCG trajectory changed.",
    ),
    metric(
        "Cache rigid edge kinematics",
        "Reduced KKT",
        33.091,
        27.522,
        "us/PCG",
        "strict",
        "58787518",
        "Combined forest expand/project",
    ),
    metric(
        "Occupancy-clamped dynamic work",
        "Quadrants runtime",
        65.347,
        36.108,
        "ms/9 frames",
        "strict",
        "quadrants@12119f03e",
        "All paired dynamic-range work",
    ),
    metric(
        "Parallel contact branches",
        "Graph scheduling",
        20.359,
        18.161,
        "ms/frame",
        "strict",
        "e7835bdd + quadrants@3b0f9c2ad",
        "Kernel interval union",
    ),
    metric(
        "Inline dynamic range bounds",
        "Quadrants lowering",
        10.340,
        0.0,
        "ms/9 frames",
        "strict",
        "quadrants@bf3a00645",
        "Five PCG helper variants, 13,030 launches",
        caveat="The numerical kernels remain; only helper launches are removed.",
    ),
]


END_TO_END_METRICS = [
    metric(
        "EE rank-1: full step",
        "Contact assembly",
        1.0,
        0.942,
        "normalized step",
        "pinned_cgq",
        "28b42c8f",
        "Pinned-CGQ A/B",
    ),
    metric(
        "Morton OneSweep: total GPU",
        "Sort",
        188.504,
        177.448,
        "ms/9 frames",
        "strict",
        "93be3590",
        "Matched profile",
    ),
    metric(
        "OneSweep/scan/FSR: total GPU",
        "Sort/reduce",
        216.494,
        190.349,
        "ms/9 frames",
        "strict",
        "ba777cda",
        "Matched profile",
    ),
    metric(
        "Fused forest: wall median",
        "Reduced KKT",
        65.399,
        53.783,
        "ms/frame",
        "strict",
        "910f9f4c",
        "20 warmup + 100 measured frames",
    ),
    metric(
        "Proxy fusion: PCG body",
        "Reduced KKT",
        169.6,
        156.4,
        "us/PCG",
        "strict",
        "910f9f4c",
        "Equal-work PCG implementation cost",
    ),
    metric(
        "Edge cache: total GPU",
        "Reduced KKT",
        220.306,
        216.494,
        "ms/9 frames",
        "strict",
        "58787518",
        "656/658 PCG iterations",
    ),
    metric(
        "Controller conformance: wall",
        "Numerical work",
        48.1,
        30.2,
        "ms/frame",
        "work_change",
        "2c2b5a1c",
        "Ten-frame median",
        caveat="Correct parameter parity reduced Newton work; not a throughput-only change.",
    ),
    metric(
        "Accept converged direction: wall",
        "Numerical work",
        34.799,
        29.465,
        "ms/frame",
        "work_change",
        "7a107b2d",
        "Controller-aligned ten-frame trace",
        caveat="Corrected accepted-step semantics and later PCG work.",
    ),
    metric(
        "Conformance + app optimization",
        "Cumulative",
        43.009,
        29.165,
        "ms/frame",
        "mixed",
        "5b6bb1ad",
        "100-frame Franka-Cloth median",
        caveat="Cumulative result; constituent changes are not additive.",
    ),
    metric(
        "Resident occupancy grid: wall",
        "Quadrants runtime",
        24.888,
        20.230,
        "ms/frame",
        "strict",
        "quadrants@12119f03e",
        "Matched frames 20-28, exactly 646 PCG iterations",
    ),
    metric(
        "Parallel contact branches: wall mean",
        "Graph scheduling",
        25.359,
        23.916,
        "ms/frame",
        "strict",
        "e7835bdd + quadrants@3b0f9c2ad",
        "Frames 20-39, 6,583/6,570 PCG iterations",
    ),
    metric(
        "Range-bound inlining: wall median",
        "Quadrants lowering",
        27.071,
        23.933,
        "ms/frame",
        "trajectory_sensitive",
        "quadrants@bf3a00645",
        "20 warmup + 200 measured cloth-stack frames",
        caveat="Raw wall result also includes a changed PCG trajectory.",
    ),
]


COMPILE_METRICS = [
    metric(
        "Disable advanced optimization",
        "Compiler configuration",
        175.64,
        54.68,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "Quadrants 1.3.1 cold backend",
        caveat="Approximately 3.5% steady-state runtime cost in this early build.",
    ),
    metric(
        "Skip inapplicable merge_global_ptrs",
        "Compiler pass",
        51.99,
        38.57,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "Same-revision O1 backend",
    ),
    metric(
        "Lower AST without SSA user scans",
        "Compiler pass",
        10.429,
        0.229,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "lower_ast pass",
    ),
    metric(
        "Indexed constant-fold replacement",
        "Compiler pass",
        5.841,
        0.527,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "constant_fold pass",
    ),
    metric(
        "Combined lowering/folding backend",
        "Compiler pipeline",
        38.57,
        22.99,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "End-to-end backend after prototypes 2 and 3",
    ),
    metric(
        "CUDA LLVM O3 to O1",
        "Backend configuration",
        22.14,
        20.04,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "Same-revision final code",
    ),
    metric(
        "Warm fastcache restart",
        "Compiler cache",
        22.08,
        0.82,
        "s",
        "strict",
        "quadrants@e5811c2dd",
        "Cold population versus unchanged-process restart",
        caveat="Cache reuse, not faster cold compilation.",
    ),
    metric(
        "Aggregate optimized compiler",
        "Compiler aggregate",
        175.64,
        28.44,
        "s",
        "cross_revision",
        "quadrants@e5811c2dd",
        "Initial 1.3.1 versus final local O3 prototype",
        caveat="Cross-revision context only; not a strict single-patch A/B.",
    ),
]


VALIDATION_CASES = [
    {
        "name": "pinned_drape",
        "vertices": 6561,
        "triangles": 12800,
        "genesis_mean_ms": 19.057,
        "cgq_mean_ms": 19.745,
        "genesis_pcg": 112804,
        "cgq_pcg": 112791,
        "genesis_newton": 648,
        "cgq_newton": 648,
        "wall_per_pcg_ratio": 0.965,
        "classification": "equal_work",
    },
    {
        "name": "inclined_drop",
        "vertices": 6561,
        "triangles": 12800,
        "genesis_mean_ms": 20.604,
        "cgq_mean_ms": 20.828,
        "genesis_pcg": 101981,
        "cgq_pcg": 100879,
        "genesis_newton": 513,
        "cgq_newton": 508,
        "wall_per_pcg_ratio": 0.979,
        "classification": "approximately_aligned",
    },
    {
        "name": "crossed_drop",
        "vertices": 5202,
        "triangles": 10000,
        "genesis_mean_ms": 34.224,
        "cgq_mean_ms": 35.240,
        "genesis_pcg": 157472,
        "cgq_pcg": 168193,
        "genesis_newton": 971,
        "cgq_newton": 1081,
        "wall_per_pcg_ratio": 1.037,
        "classification": "trajectory_divergent",
    },
    {
        "name": "multilayer",
        "vertices": 8804,
        "triangles": 17000,
        "genesis_mean_ms": 24.938,
        "cgq_mean_ms": 23.095,
        "genesis_pcg": 103919,
        "cgq_pcg": 96749,
        "genesis_newton": 500,
        "cgq_newton": 500,
        "wall_per_pcg_ratio": 1.005,
        "classification": "coarsely_normalized",
    },
]


REPORT_DATA = {
    "title": "Coupling Performance Optimization Report",
    "generated": "2026-09-30",
    "hardware": "NVIDIA GeForce RTX 5090",
    "runtime_commit": "7f8850d4",
    "native_reference_commit": "42e7d4cbbad08739107ad830a17918f5f0f209ff",
    "compiler_commits": {
        "compile_time": "e5811c2dd",
        "checkpoint_parallel": "3b0f9c2ad",
        "resident_grid": "12119f03e",
        "range_bound_inlining": "bf3a00645",
    },
    "compiler_publication": {
        "upstream_status": "Not merged; only the compile-time subset has an open synchronization PR.",
    },
    "artifact_limitations": [
        ("JSON/CSV summaries and benchmark scripts are retained, but not " "every original Nsight SQLite capture."),
        ("Named-kernel figures whose raw database is absent require " "rerunning the documented profile window."),
    ],
    "stage_metrics": STAGE_METRICS,
    "end_to_end_metrics": END_TO_END_METRICS,
    "compile_metrics": COMPILE_METRICS,
    "validation_cases": VALIDATION_CASES,
    "non_acceleration_result": {
        "compile_threads_4_s": 174.67,
        "compile_threads_24_s": 175.64,
        "conclusion": "24 threads did not accelerate the serial pre-offload bottleneck.",
    },
}


COLORS = {
    "strict": "#2563eb",
    "pinned_cgq": "#7c3aed",
    "work_change": "#d97706",
    "mixed": "#0f766e",
    "trajectory_sensitive": "#64748b",
    "cross_revision": "#9f1239",
}


def add_bar_labels(ax: plt.Axes, bars, values: list[float], suffix: str) -> None:
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.015 * max(values),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}{suffix}",
            va="center",
            fontsize=8,
        )


def plot_stage_reductions(ax: plt.Axes) -> None:
    selected_names = {
        "Remove duplicate native collision",
        "Warp-batched EE query/output",
        "Fused rigid forest P/P^T",
        "Native-f32 scene bounds",
        "Fused proxy map/post",
        "LBVH Morton OneSweep",
        "OneSweep + scan + warp FSR",
        "Load-balanced dual EE",
        "Warp-cooperative PT",
        "Direct internal refit",
        "Packed/direct broad phase",
        "Cache rigid edge kinematics",
    }
    items = [item for item in STAGE_METRICS if item["name"] in selected_names]
    items.sort(key=lambda item: float(item["reduction_percent"] or 0.0))
    labels = [str(item["name"]) for item in items]
    values = [float(item["reduction_percent"] or 0.0) for item in items]
    colors = [COLORS[str(item["evidence"])] for item in items]
    bars = ax.barh(labels, values, color=colors)
    add_bar_labels(ax, bars, values, "%")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Stage cost reduction (%)")
    ax.set_ylabel("Optimization")
    ax.set_title("A. Isolated stage improvements")
    ax.grid(axis="x", alpha=0.25)


def plot_end_to_end(ax: plt.Axes) -> None:
    selected_names = {
        "Morton OneSweep: total GPU",
        "OneSweep/scan/FSR: total GPU",
        "Fused forest: wall median",
        "Controller conformance: wall",
        "Accept converged direction: wall",
        "Resident occupancy grid: wall",
        "Parallel contact branches: wall mean",
        "Range-bound inlining: wall median",
        "Conformance + app optimization",
    }
    items = [item for item in END_TO_END_METRICS if item["name"] in selected_names]
    items.sort(key=lambda item: float(item["speedup"] or 0.0))
    labels = [str(item["name"]).replace(": ", "\n") for item in items]
    values = [float(item["speedup"] or 0.0) for item in items]
    colors = [COLORS[str(item["evidence"])] for item in items]
    bars = ax.barh(labels, values, color=colors)
    add_bar_labels(ax, bars, values, "x")
    ax.axvline(1.0, color="#111827", linewidth=0.8)
    ax.set_xlim(0.95, max(values) * 1.13)
    ax.set_xlabel("Observed speedup (before / after)")
    ax.set_ylabel("Optimization and measured scope")
    ax.set_title("B. End-to-end or complete-loop impact")
    ax.grid(axis="x", alpha=0.25)


def plot_compile_time(ax: plt.Axes) -> None:
    items = COMPILE_METRICS
    labels = [
        str(item["name"])
        .replace(" optimization", " opt.")
        .replace("inapplicable ", "")
        .replace("replacement", "replace")
        for item in items
    ]
    before = [float(item["before"] or 0.0) for item in items]
    after = [float(item["after"] or 0.0) for item in items]
    positions = list(range(len(items)))
    width = 0.36
    ax.bar(
        [position - width / 2 for position in positions],
        before,
        width,
        label="Before",
        color="#94a3b8",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        after,
        width,
        label="After",
        color="#2563eb",
    )
    for position, item in zip(positions, items, strict=True):
        speedup = float(item["speedup"] or 0.0)
        ax.text(
            position,
            max(float(item["before"] or 0.0), float(item["after"] or 0.0)) * 1.22,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_xticks(positions, labels, rotation=22, ha="right")
    ax.set_ylabel("Cold/load latency (seconds, log scale)")
    ax.set_xlabel("Compiler or cache optimization")
    ax.set_title("C. Compilation and restart latency")
    ax.legend(frameon=False)
    ax.grid(axis="y", which="both", alpha=0.25)


def write_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(18, 17), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.2, 1.0))
    plot_stage_reductions(figure.add_subplot(grid[0, 0]))
    plot_end_to_end(figure.add_subplot(grid[0, 1]))
    plot_compile_time(figure.add_subplot(grid[1, :]))
    figure.suptitle(
        "Coupling runtime: measured optimization impact",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.977,
        "RTX 5090 · Genesis 7f8850d4 · CGQ 42e7d4cb · Quadrants through bf3a00645",
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.006,
        "Scopes and baselines differ; speedups are not additive. "
        "Purple = pinned CGQ migration evidence, blue = strict matched A/B, "
        "orange = numerical-work correction, teal = cumulative mixed, gray = trajectory-sensitive.",
        ha="center",
        fontsize=9,
    )
    legend = [
        Patch(color=COLORS["strict"], label="Strict matched A/B"),
        Patch(color=COLORS["pinned_cgq"], label="Pinned CGQ migration evidence"),
        Patch(color=COLORS["work_change"], label="Correctness / numerical-work change"),
        Patch(color=COLORS["mixed"], label="Cumulative mixed sequence"),
        Patch(color=COLORS["trajectory_sensitive"], label="Trajectory-sensitive aggregate"),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=5,
        frameon=False,
    )
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_data() -> None:
    DATA_PATH.write_text(
        json.dumps(REPORT_DATA, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_bundle() -> None:
    required = (REPORT_PATH, DATA_PATH, PLOT_PATH, Path(__file__))
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot package missing report assets: {missing}")
    with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in required:
            archive.write(path, arcname=f"coupling-performance-report/{path.name}")


def main() -> None:
    write_data()
    write_plot()
    if REPORT_PATH.exists():
        write_bundle()
    stage_count = len(STAGE_METRICS)
    compile_count = len(COMPILE_METRICS)
    print(
        f"Generated {PLOT_PATH.name}, {DATA_PATH.name}, "
        f"{stage_count} runtime-stage metrics, and {compile_count} compile metrics."
    )
    if BUNDLE_PATH.exists():
        size_mib = BUNDLE_PATH.stat().st_size / math.pow(1024.0, 2)
        print(f"Packaged {BUNDLE_PATH.name} ({size_mib:.2f} MiB).")


if __name__ == "__main__":
    main()
