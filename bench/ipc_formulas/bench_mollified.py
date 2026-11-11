"""Compile time and per-pair cost of the mollified edge-edge barrier path against the plain one, on the Genesis side.

    python bench/ipc_formulas/bench_mollified.py [--pairs 500000] [--repeats 20]

Needs the library worktree on PYTHONPATH, a fresh offline cache (QD_OFFLINE_CACHE_FILE_PATH) for the compile times to
mean anything, and an exclusive GPU lease for the run times. The rows are the real edge-edge pairs of cloth_stack
(inputs.py), the mollified ones for the mollified path and the others for the plain one, tiled to '--pairs' pairs. Each
kernel writes the gradient and the Hessian of every pair, as the assembly would. Prints the first call of each kernel
(compile and run) and the median run time per pair over '--repeats' calls, fast_math off and float64 on CUDA.
"""

import argparse
import time

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.ipc.barrier import (
    func_barrier_gradient_hessian,
    func_edge_edge_mollifier_threshold,
    func_mollified_barrier_gradient_hessian,
)
from genesis.engine.ipc.distance import func_edge_edge_distance_flag, func_edge_edge_flagged_distance2

from cases import (
    DATA,
    EE_PARALLEL_TOLERANCE,
    HESSIAN_CLAMP_RATIO,
    MOLLIFIER_RATIO,
    PROJECTION_EIGEN_TOLERANCE,
    PROJECTION_MAX_SWEEPS,
    PROJECTION_RANK_TOLERANCE,
)
from genesis_side import put, put_matrix, v3


@qd.kernel
def k_plain(rows: qd.types.ndarray(), out: qd.types.ndarray(), parallel_tolerance: float, hessian_clamp_ratio: float):
    for i_r in range(rows.shape[0]):
        ea0, ea1, eb0, eb1 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        flag = func_edge_edge_distance_flag(ea0, ea1, eb0, eb1, parallel_tolerance)
        distance2, distance2_gradient = func_edge_edge_flagged_distance2(flag, ea0, ea1, eb0, eb1)
        gradient, hessian = func_barrier_gradient_hessian(
            distance2, distance2_gradient, rows[i_r, 24], rows[i_r, 25], rows[i_r, 26], hessian_clamp_ratio
        )
        put(out, i_r, 0, gradient)
        put_matrix(out, i_r, 12, hessian)


@qd.kernel
def k_mollified(
    rows: qd.types.ndarray(),
    out: qd.types.ndarray(),
    parallel_tolerance: float,
    mollifier_ratio: float,
    rank_tolerance: float,
    eigen_tolerance: float,
    max_sweeps: int,
):
    for i_r in range(rows.shape[0]):
        ea0, ea1, eb0, eb1 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        flag = func_edge_edge_distance_flag(ea0, ea1, eb0, eb1, parallel_tolerance)
        distance2, distance2_gradient = func_edge_edge_flagged_distance2(flag, ea0, ea1, eb0, eb1)
        threshold = func_edge_edge_mollifier_threshold(
            v3(rows, i_r, 12), v3(rows, i_r, 15), v3(rows, i_r, 18), v3(rows, i_r, 21), mollifier_ratio
        )
        gradient, hessian = func_mollified_barrier_gradient_hessian(
            flag,
            ea0,
            ea1,
            eb0,
            eb1,
            distance2,
            distance2_gradient,
            threshold,
            rows[i_r, 24],
            rows[i_r, 25],
            rows[i_r, 26],
            rank_tolerance,
            eigen_tolerance,
            max_sweeps,
        )
        put(out, i_r, 0, gradient)
        put_matrix(out, i_r, 12, hessian)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    qd.lang.impl.current_cfg().fast_math = False
    with np.load(DATA / "inputs.npz") as inputs:
        mollified = inputs["barrier_ee/real_cloth_stack_mollified"]
        rows = inputs["barrier_ee/real_cloth_stack"]
    rest_a, rest_b = rows[:, 15:18] - rows[:, 12:15], rows[:, 21:24] - rows[:, 18:21]
    cross = np.cross(rows[:, 3:6] - rows[:, 0:3], rows[:, 9:12] - rows[:, 6:9])
    is_mollified = (cross * cross).sum(1) < MOLLIFIER_RATIO * (rest_a * rest_a).sum(1) * (rest_b * rest_b).sum(1)
    plain = rows[~is_mollified]
    out = torch.empty((args.pairs, 12 + 144), dtype=gs.tc_float, device=gs.device)
    runs = {
        "plain": (plain, lambda r: k_plain(r, out, EE_PARALLEL_TOLERANCE, HESSIAN_CLAMP_RATIO)),
        "mollified": (
            mollified,
            lambda r: k_mollified(
                r,
                out,
                EE_PARALLEL_TOLERANCE,
                MOLLIFIER_RATIO,
                PROJECTION_RANK_TOLERANCE,
                PROJECTION_EIGEN_TOLERANCE,
                PROJECTION_MAX_SWEEPS,
            ),
        ),
    }
    for name, (source, run) in runs.items():
        tiled = np.resize(source, (args.pairs, source.shape[1]))
        rows_t = torch.as_tensor(tiled, dtype=gs.tc_float, device=gs.device).contiguous()
        torch.cuda.synchronize()
        start = time.perf_counter()
        run(rows_t)
        torch.cuda.synchronize()
        first = time.perf_counter() - start
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            run(rows_t)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        median = float(np.median(times))
        print(
            f"{name}: {len(source)} distinct pairs tiled to {args.pairs}; first call (compile + run) {first:.2f} s; "
            f"median run {median * 1e3:.2f} ms, {median / args.pairs * 1e9:.1f} ns per pair",
            flush=True,
        )


if __name__ == "__main__":
    main()
