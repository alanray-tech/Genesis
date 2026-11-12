"""Time the barrier of one edge-edge pair on the GPU, per variant, with the Genesis funcs the contact assembly calls.

Variants, each thread taking a slightly shifted copy of a near-parallel pair in the contact band (d_hat 1e-2):
- rank1: the barrier of a regular pair (func_barrier_gradient_hessian), the path of most pairs;
- mollified_no_projection: the product-rule gradient and Hessian of func_mollified_barrier_gradient_hessian without its
  projection (the formula of that func, repeated here);
- mollified: func_mollified_barrier_gradient_hessian itself, projection onto the PSD cone included (qd_project_psd);
- eigen12: qd_symmetric_eigen of a full symmetric 12x12, the path of a Hessian above the low-rank limit.
One thread gives the latency of one pair, which bounds a launch holding a single such pair; many threads the throughput.

    python mollified_latency.py [--repeats 20]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import quadrants as qd

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=20)
args = parser.parse_args()

gs.init(backend=gs.gpu, precision="64", logging_level="warning")
import genesis.utils.geom as gu  # noqa: E402
from genesis.engine.ipc.barrier import (  # noqa: E402
    func_barrier,
    func_barrier_gradient_hessian,
    func_mollified_barrier_gradient_hessian,
)
from genesis.engine.ipc.distance import (  # noqa: E402
    func_edge_edge_cross_norm2,
    func_edge_edge_distance_flag,
    func_edge_edge_flagged_distance2,
    func_edge_edge_flagged_distance2_hessian,
)

D_HAT, THICKNESS, KAPPA = 1e-2, 2e-3, 1e5
REST_LENGTH2 = 0.038**2
THRESHOLD = 1e-3 * REST_LENGTH2 * REST_LENGTH2
N_MAX = 1 << 15
out = qd.field(gs.qd_float, shape=N_MAX)


@qd.func
def func_pair(i: int):
    shift = 1e-7 * i
    ea0 = qd.Vector([0.0, 0.0, shift], dt=gs.qd_float)
    ea1 = qd.Vector([0.038, 0.0, shift], dt=gs.qd_float)
    eb0 = qd.Vector([0.001, 0.005 + shift, 0.0], dt=gs.qd_float)
    eb1 = qd.Vector([0.039, 0.005 + shift, 5e-5], dt=gs.qd_float)
    flag = func_edge_edge_distance_flag(ea0, ea1, eb0, eb1, 1e-6)
    distance2, distance2_gradient = func_edge_edge_flagged_distance2(flag, ea0, ea1, eb0, eb1)
    return ea0, ea1, eb0, eb1, flag, distance2, distance2_gradient


@qd.kernel
def kernel_rank1(n: int):
    for i in range(n):
        _ea0, _ea1, _eb0, _eb1, _flag, distance2, distance2_gradient = func_pair(i)
        gradient, hessian = func_barrier_gradient_hessian(distance2, distance2_gradient, D_HAT, THICKNESS, KAPPA, 1e-6)
        out[i] = gradient.sum() + hessian.sum()


@qd.kernel
def kernel_mollified_no_projection(n: int):
    for i in range(n):
        ea0, ea1, eb0, eb1, flag, distance2, distance2_gradient = func_pair(i)
        cross_norm2, cross_norm2_gradient, cross_norm2_hessian = func_edge_edge_cross_norm2(ea0, ea1, eb0, eb1)
        distance2_hessian = func_edge_edge_flagged_distance2_hessian(flag, ea0, ea1, eb0, eb1)
        energy, first, second = func_barrier(distance2, D_HAT, THICKNESS, KAPPA)
        inverse_threshold = 1.0 / THRESHOLD
        mollifier = -(cross_norm2 * cross_norm2) * inverse_threshold * inverse_threshold + (
            2.0 * cross_norm2 * inverse_threshold
        )
        mollifier_first = -2.0 * cross_norm2 * inverse_threshold * inverse_threshold + 2.0 * inverse_threshold
        mollifier_second = -2.0 * inverse_threshold * inverse_threshold
        mollifier_gradient = mollifier_first * cross_norm2_gradient
        mollifier_hessian = (
            mollifier_second * cross_norm2_gradient.outer_product(cross_norm2_gradient)
            + mollifier_first * cross_norm2_hessian
        )
        gradient = energy * mollifier_gradient + (mollifier * first) * distance2_gradient
        coupling = mollifier_gradient.outer_product(distance2_gradient)
        hessian = (
            energy * mollifier_hessian
            + first * (coupling + coupling.transpose())
            + mollifier * (second * distance2_gradient.outer_product(distance2_gradient) + first * distance2_hessian)
        )
        out[i] = gradient.sum() + hessian.sum()


@qd.kernel
def kernel_mollified(n: int):
    for i in range(n):
        ea0, ea1, eb0, eb1, flag, distance2, distance2_gradient = func_pair(i)
        gradient, hessian = func_mollified_barrier_gradient_hessian(
            flag, ea0, ea1, eb0, eb1, distance2, distance2_gradient, THRESHOLD, D_HAT, THICKNESS, KAPPA, 1e-12, 1e-15, 12
        )
        out[i] = gradient.sum() + hessian.sum()


@qd.kernel
def kernel_eigen12(n: int):
    for i in range(n):
        matrix = qd.Matrix.zero(gs.qd_float, 12, 12)
        for r in qd.static(range(12)):
            for c in qd.static(range(12)):
                matrix[r, c] = 1.0 / (1.0 + r + c + 1e-6 * i) + qd.select(r == c, 2.0, 0.0)
        eigenvalues, eigenvectors = gu.qd_symmetric_eigen(matrix, 1e-15, 12)
        out[i] = eigenvalues.sum() + eigenvectors.sum()


results = {}
for name, kernel in (
    ("rank1", kernel_rank1),
    ("mollified_no_projection", kernel_mollified_no_projection),
    ("mollified", kernel_mollified),
    ("eigen12", kernel_eigen12),
):
    kernel(1)
    qd.sync()
    results[name] = {}
    for n in (1, 32, 1024, N_MAX):
        kernel(n)
        qd.sync()
        start = time.perf_counter()
        for _ in range(args.repeats):
            kernel(n)
        qd.sync()
        results[name][n] = 1e6 * (time.perf_counter() - start) / args.repeats
    print(f"{name}: " + ", ".join(f"{n} threads {us:.1f} us" for n, us in results[name].items()), flush=True)
Path(__file__).parent.joinpath("results", "mollified_latency.json").write_text(json.dumps(results, indent=1))
print(json.dumps({name: float(np.round(r[1], 1)) for name, r in results.items()}))
