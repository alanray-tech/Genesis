"""Genesis side of the IPC formula parity: the functions of genesis.engine.ipc on the input rows, in Quadrants kernels.

    python bench/ipc_formulas/genesis_side.py --backend cuda|cpu --fast-math true|false

Needs the library worktree on PYTHONPATH (genesis.engine.ipc). Reads data/inputs.npz (inputs.py) and writes
data/out_genesis_<backend>_fast-math-<bool>.npz, one array per case and input set as cgq_side.py does. '--fast-math'
is the global compile flag of Quadrants, which Genesis turns on outside debug mode: it lets the compiler contract
multiplies and adds into fused multiply-adds and reassociate them. It is switched off after gs.init, before any kernel
compiles, so give each setting its own offline cache (QD_OFFLINE_CACHE_FILE_PATH).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import quadrants as qd
from quadrants.lang import matrix_ops
from quadrants.lang.kernel_impl import pyfunc
from quadrants.types.annotations import template

import genesis as gs
from genesis.engine.ipc.barrier import (
    func_barrier,
    func_barrier_gradient_hessian,
    func_barrier_normal_force,
    func_edge_edge_mollifier_threshold,
    func_mollified_barrier,
    func_mollified_barrier_gradient_hessian,
    func_pair_barrier_parameters,
    func_point_halfplane_barrier_gradient_hessian,
)
from genesis.engine.ipc.ccd import (
    func_additive_ccd,
    func_candidate_reach,
    func_directional_ccd,
    func_point_halfplane_candidate_reach,
    func_point_halfplane_ccd,
    func_point_halfplane_screw_ccd,
    func_screw_ccd,
)
from genesis.engine.ipc.distance import (
    func_closest_points,
    func_edge_edge_closest_points,
    func_edge_edge_cross_norm2,
    func_edge_edge_distance2,
    func_edge_edge_distance_flag,
    func_edge_edge_distance_type,
    func_edge_edge_flagged_distance2,
    func_point_edge_closest_point,
    func_point_edge_distance2,
    func_point_edge_distance_flag,
    func_point_edge_flagged_distance2,
    func_point_triangle_closest_point,
    func_point_triangle_distance2,
    func_point_triangle_distance_flag,
    func_point_triangle_flagged_distance2,
)
from genesis.engine.ipc.friction import (
    func_edge_edge_friction_frame,
    func_friction_energy,
    func_friction_gradient_hessian,
    func_point_edge_friction_frame,
    func_point_halfplane_friction_frame,
    func_point_point_friction_frame,
    func_point_triangle_friction_frame,
)
from genesis.engine.ipc.intersection import func_edge_triangle_intersect

from cases import (
    CANDIDATE_REACH_SCALE,
    CASES,
    CCD_MAX_ITERS,
    EE_PARALLEL_TOLERANCE,
    EE_TYPE_PARALLEL_TOLERANCE,
    HESSIAN_CLAMP_RATIO,
    INTERSECTION_TOLERANCE,
    MOLLIFIER_RATIO,
    STENCIL_VERTS,
    n_outputs,
)

HERE = Path(__file__).resolve().parent


@pyfunc
def _reduce_eigen_order(mat, fun: template()):
    """Quadrants' reduction of a tensor, 3- and 4-vectors being summed pairwise in Eigen's order.

    Eigen's fixed-size reductions (dot, squaredNorm) split the vector in halves, x0 + (x1 + x2) for three entries, where
    Quadrants sums from the left, (x0 + x1) + x2. Experiment only: '--sum-order eigen' installs it in place of
    quadrants.lang.matrix_ops._reduce, which every dot product, squared norm and sum of the library goes through.
    """
    shape = qd.static(mat.get_shape())
    if qd.static(len(shape) == 1 and shape[0] == 3):
        return fun(mat[0], fun(mat[1], mat[2]))
    if qd.static(len(shape) == 1 and shape[0] == 4):
        return fun(fun(mat[0], mat[1]), fun(mat[2], mat[3]))
    return _reduce_quadrants_order(mat, fun)


_reduce_quadrants_order = matrix_ops._reduce


@qd.func
def v3(rows: qd.template(), i_r: int, col: qd.template()):
    return qd.Vector([rows[i_r, col], rows[i_r, col + 1], rows[i_r, col + 2]], dt=gs.qd_float)


@qd.func
def points(rows: qd.template(), i_r: int, col: qd.template(), n: qd.template()):
    return qd.Matrix(
        [[rows[i_r, col + 3 * i + k] for k in qd.static(range(3))] for i in qd.static(range(n))], dt=gs.qd_float
    )


@qd.func
def scalars(rows: qd.template(), i_r: int, col: qd.template(), n: qd.template()):
    return qd.Vector([rows[i_r, col + i] for i in qd.static(range(n))], dt=gs.qd_float)


@qd.func
def put(out: qd.template(), i_r: int, col: qd.template(), value):
    for k in qd.static(range(value.n)):
        out[i_r, col + k] = value[k]


@qd.func
def put_matrix(out: qd.template(), i_r: int, col: qd.template(), value):
    for i in qd.static(range(value.n)):
        for j in qd.static(range(value.m)):
            out[i_r, col + i * value.m + j] = value[i, j]


# --- Distances ------------------------------------------------------------------------------------------------------


@qd.kernel
def k_pt_distance(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        p, t0, t1, t2 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        flag = func_point_triangle_distance_flag(p, t0, t1, t2)
        distance2, gradient = func_point_triangle_flagged_distance2(flag, p, t0, t1, t2)
        out[i_r, 0] = flag
        out[i_r, 1] = distance2
        put(out, i_r, 2, gradient)


@qd.kernel
def k_ee_distance(rows: qd.types.ndarray(), out: qd.types.ndarray(), parallel_tolerance: float):
    for i_r in range(rows.shape[0]):
        ea0, ea1, eb0, eb1 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        flag = func_edge_edge_distance_flag(ea0, ea1, eb0, eb1, parallel_tolerance)
        distance2, gradient = func_edge_edge_flagged_distance2(flag, ea0, ea1, eb0, eb1)
        out[i_r, 0] = flag
        out[i_r, 1] = distance2
        put(out, i_r, 2, gradient)


@qd.kernel
def k_pe_distance(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        p, e0, e1 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6)
        flag = func_point_edge_distance_flag(p, e0, e1)
        distance2, gradient = func_point_edge_flagged_distance2(flag, p, e0, e1)
        out[i_r, 0] = flag
        out[i_r, 1] = distance2
        put(out, i_r, 2, gradient)


@qd.kernel
def k_ee_distance_type(rows: qd.types.ndarray(), out: qd.types.ndarray(), parallel_tolerance: float):
    for i_r in range(rows.shape[0]):
        out[i_r, 0] = func_edge_edge_distance_type(
            v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9), parallel_tolerance
        )


@qd.kernel
def k_closest_pt(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        p = v3(rows, i_r, 0)
        distance, closest = func_point_triangle_closest_point(p, v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9))
        out[i_r, 0] = distance
        put(out, i_r, 1, p)
        put(out, i_r, 4, closest)


@qd.kernel
def k_closest_ee(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        distance, closest_a, closest_b = func_edge_edge_closest_points(
            v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        )
        out[i_r, 0] = distance
        put(out, i_r, 1, closest_a)
        put(out, i_r, 4, closest_b)


@qd.kernel
def k_closest_pe(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        p = v3(rows, i_r, 0)
        distance, closest = func_point_edge_closest_point(p, v3(rows, i_r, 3), v3(rows, i_r, 6))
        out[i_r, 0] = distance
        put(out, i_r, 1, p)
        put(out, i_r, 4, closest)


# --- Barrier --------------------------------------------------------------------------------------------------------


@qd.kernel
def k_barrier(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        energy, first, second = func_barrier(rows[i_r, 0], rows[i_r, 1], rows[i_r, 2], rows[i_r, 3])
        out[i_r, 0] = energy
        out[i_r, 1] = first
        out[i_r, 2] = second
        out[i_r, 3] = func_barrier_normal_force(rows[i_r, 0], rows[i_r, 1], rows[i_r, 2], rows[i_r, 3])


@qd.kernel
def k_mollified_barrier(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        out[i_r, 0] = func_mollified_barrier(
            rows[i_r, 0], rows[i_r, 1], rows[i_r, 2], rows[i_r, 3], rows[i_r, 4], rows[i_r, 5]
        )


@qd.kernel
def k_mollifier(rows: qd.types.ndarray(), out: qd.types.ndarray(), mollifier_ratio: float):
    for i_r in range(rows.shape[0]):
        x0, x1, x2, x3 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        out[i_r, 0] = func_edge_edge_mollifier_threshold(x0, x1, x2, x3, mollifier_ratio)
        cross_norm2, _ = func_edge_edge_cross_norm2(x0, x1, x2, x3)
        out[i_r, 1] = cross_norm2


def make_pair_kernel(n_a, n_b):
    @qd.kernel
    def k_pair(rows: qd.types.ndarray(), out: qd.types.ndarray()):
        for i_r in range(rows.shape[0]):
            d_hat, thickness, area_weight = func_pair_barrier_parameters(
                rows[i_r, 0],
                rows[i_r, 1],
                scalars(rows, i_r, 2, n_a),
                scalars(rows, i_r, 2 + n_a, n_b),
                rows[i_r, 6],
                rows[i_r, 7],
            )
            out[i_r, 0] = d_hat
            out[i_r, 1] = thickness
            out[i_r, 2] = area_weight

    return k_pair


@qd.kernel
def k_barrier_pt(rows: qd.types.ndarray(), out: qd.types.ndarray(), hessian_clamp_ratio: float):
    for i_r in range(rows.shape[0]):
        p, t0, t1, t2 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        flag = func_point_triangle_distance_flag(p, t0, t1, t2)
        distance2, distance2_gradient = func_point_triangle_flagged_distance2(flag, p, t0, t1, t2)
        gradient, hessian = func_barrier_gradient_hessian(
            distance2, distance2_gradient, rows[i_r, 12], rows[i_r, 13], rows[i_r, 14], hessian_clamp_ratio
        )
        out[i_r, 0] = flag
        put(out, i_r, 1, gradient)
        put_matrix(out, i_r, 13, hessian)


@qd.kernel
def k_barrier_ee(
    rows: qd.types.ndarray(),
    out: qd.types.ndarray(),
    parallel_tolerance: float,
    mollifier_ratio: float,
    hessian_clamp_ratio: float,
):
    for i_r in range(rows.shape[0]):
        ea0, ea1, eb0, eb1 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        d_hat, thickness, kappa = rows[i_r, 24], rows[i_r, 25], rows[i_r, 26]
        flag = func_edge_edge_distance_flag(ea0, ea1, eb0, eb1, parallel_tolerance)
        distance2, distance2_gradient = func_edge_edge_flagged_distance2(flag, ea0, ea1, eb0, eb1)
        cross_norm2, cross_norm2_gradient = func_edge_edge_cross_norm2(ea0, ea1, eb0, eb1)
        threshold = func_edge_edge_mollifier_threshold(
            v3(rows, i_r, 12), v3(rows, i_r, 15), v3(rows, i_r, 18), v3(rows, i_r, 21), mollifier_ratio
        )
        gradient = qd.Vector.zero(gs.qd_float, 12)
        hessian = qd.Matrix.zero(gs.qd_float, 12, 12)
        is_mollified = cross_norm2 < threshold
        if is_mollified:
            gradient, hessian = func_mollified_barrier_gradient_hessian(
                distance2,
                distance2_gradient,
                cross_norm2,
                cross_norm2_gradient,
                threshold,
                d_hat,
                thickness,
                kappa,
                hessian_clamp_ratio,
            )
        else:
            gradient, hessian = func_barrier_gradient_hessian(
                distance2, distance2_gradient, d_hat, thickness, kappa, hessian_clamp_ratio
            )
        out[i_r, 0] = flag
        out[i_r, 1] = qd.select(is_mollified, 1.0, 0.0)
        put(out, i_r, 2, gradient)
        put_matrix(out, i_r, 14, hessian)


@qd.kernel
def k_barrier_pe(rows: qd.types.ndarray(), out: qd.types.ndarray(), hessian_clamp_ratio: float):
    for i_r in range(rows.shape[0]):
        p, e0, e1 = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6)
        flag = func_point_edge_distance_flag(p, e0, e1)
        distance2, distance2_gradient = func_point_edge_flagged_distance2(flag, p, e0, e1)
        gradient, hessian = func_barrier_gradient_hessian(
            distance2, distance2_gradient, rows[i_r, 9], rows[i_r, 10], rows[i_r, 11], hessian_clamp_ratio
        )
        out[i_r, 0] = flag
        put(out, i_r, 1, gradient)
        put_matrix(out, i_r, 10, hessian)


@qd.kernel
def k_barrier_pp(rows: qd.types.ndarray(), out: qd.types.ndarray(), hessian_clamp_ratio: float):
    for i_r in range(rows.shape[0]):
        offset = v3(rows, i_r, 0) - v3(rows, i_r, 3)
        distance2_gradient = qd.Vector([2.0 * offset[k % 3] * (1.0 - 2.0 * (k // 3)) for k in qd.static(range(6))])
        gradient, hessian = func_barrier_gradient_hessian(
            offset.norm_sqr(), distance2_gradient, rows[i_r, 6], rows[i_r, 7], rows[i_r, 8], hessian_clamp_ratio
        )
        put(out, i_r, 0, gradient)
        put_matrix(out, i_r, 6, hessian)


@qd.kernel
def k_barrier_ph(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        normal = v3(rows, i_r, 6)
        gradient, hessian = func_point_halfplane_barrier_gradient_hessian(
            normal.dot(v3(rows, i_r, 0) - v3(rows, i_r, 3)), normal, rows[i_r, 9], rows[i_r, 10], rows[i_r, 11]
        )
        put(out, i_r, 0, gradient)
        put_matrix(out, i_r, 3, hessian)


# --- Friction -------------------------------------------------------------------------------------------------------


def make_friction_kernel(kind):
    n = sum(STENCIL_VERTS[kind])

    @qd.kernel
    def k_friction(rows: qd.types.ndarray(), out: qd.types.ndarray()):
        for i_r in range(rows.shape[0]):
            current = points(rows, i_r, 0, n)
            lagged = points(rows, i_r, 3 * n, n)
            d_hat, kappa, mu = rows[i_r, 6 * n], rows[i_r, 6 * n + 1], rows[i_r, 6 * n + 2]
            eps_vh, thickness = rows[i_r, 6 * n + 3], rows[i_r, 6 * n + 4]
            distance2 = gs.qd_float(0.0)
            basis = qd.Matrix.zero(gs.qd_float, 3, 2)
            weights = qd.Vector.zero(gs.qd_float, n)
            if qd.static(kind == "pt"):
                distance2 = func_point_triangle_distance2(lagged[0, :], lagged[1, :], lagged[2, :], lagged[3, :])
                basis, weights = func_point_triangle_friction_frame(
                    lagged[0, :], lagged[1, :], lagged[2, :], lagged[3, :]
                )
            elif qd.static(kind == "ee"):
                distance2 = func_edge_edge_distance2(lagged[0, :], lagged[1, :], lagged[2, :], lagged[3, :])
                basis, weights = func_edge_edge_friction_frame(lagged[0, :], lagged[1, :], lagged[2, :], lagged[3, :])
            elif qd.static(kind == "pe"):
                distance2 = func_point_edge_distance2(lagged[0, :], lagged[1, :], lagged[2, :])
                basis, weights = func_point_edge_friction_frame(lagged[0, :], lagged[1, :], lagged[2, :])
            else:
                distance2 = (lagged[0, :] - lagged[1, :]).norm_sqr()
                basis, weights = func_point_point_friction_frame(lagged[0, :], lagged[1, :])
            normal_force = func_barrier_normal_force(distance2, d_hat, thickness, kappa)
            displacements = current - lagged
            out[i_r, 0] = func_friction_energy(basis, weights, displacements, normal_force, mu, eps_vh)
            gradient, hessian = func_friction_gradient_hessian(basis, weights, displacements, normal_force, mu, eps_vh)
            put(out, i_r, 1, gradient)
            put_matrix(out, i_r, 1 + 3 * n, hessian)

    return k_friction


@qd.kernel
def k_friction_ph(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        x, lagged, plane_point, normal = v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9)
        d_hat, kappa, mu, eps_vh, thickness = rows[i_r, 12], rows[i_r, 13], rows[i_r, 14], rows[i_r, 15], rows[i_r, 16]
        distance = normal.dot(lagged - plane_point)
        normal_force = func_barrier_normal_force(distance * distance, d_hat, thickness, kappa)
        basis, weights = func_point_halfplane_friction_frame(normal)
        displacements = qd.Matrix.rows([x - lagged])
        out[i_r, 0] = func_friction_energy(basis, weights, displacements, normal_force, mu, eps_vh)
        gradient, hessian = func_friction_gradient_hessian(basis, weights, displacements, normal_force, mu, eps_vh)
        put(out, i_r, 1, gradient)
        put_matrix(out, i_r, 4, hessian)


def make_frame_kernel(kind):
    n = 1 if kind == "ph" else sum(STENCIL_VERTS[kind])

    @qd.kernel
    def k_frame(rows: qd.types.ndarray(), out: qd.types.ndarray()):
        for i_r in range(rows.shape[0]):
            x = points(rows, i_r, 0, n)
            basis = qd.Matrix.zero(gs.qd_float, 3, 2)
            weights = qd.Vector.zero(gs.qd_float, max(n, 1))
            if qd.static(kind == "pt"):
                basis, weights = func_point_triangle_friction_frame(x[0, :], x[1, :], x[2, :], x[3, :])
            elif qd.static(kind == "ee"):
                basis, weights = func_edge_edge_friction_frame(x[0, :], x[1, :], x[2, :], x[3, :])
            elif qd.static(kind == "pe"):
                basis, weights = func_point_edge_friction_frame(x[0, :], x[1, :], x[2, :])
            elif qd.static(kind == "pp"):
                basis, weights = func_point_point_friction_frame(x[0, :], x[1, :])
            else:
                basis, weights = func_point_halfplane_friction_frame(x[0, :])
            if qd.static(kind in ("pp", "ph")):
                put_matrix(out, i_r, 0, basis @ basis.transpose())
            else:
                put_matrix(out, i_r, 0, basis.transpose())
                put(out, i_r, 6, weights)

    return k_frame


# --- CCD ------------------------------------------------------------------------------------------------------------


def make_ccd_linear_kernel(kind, is_directional):
    n_a, n_b = STENCIL_VERTS[kind]
    n = n_a + n_b

    @qd.kernel
    def k_ccd(rows: qd.types.ndarray(), out: qd.types.ndarray(), max_iters: int, parallel_tolerance: float):
        for i_r in range(rows.shape[0]):
            points_a, points_b = points(rows, i_r, 0, n_a), points(rows, i_r, 3 * n_a, n_b)
            displacements_a, displacements_b = points(rows, i_r, 3 * n, n_a), points(rows, i_r, 3 * n + 3 * n_a, n_b)
            eta, thickness = rows[i_r, 6 * n], rows[i_r, 6 * n + 1]
            if qd.static(is_directional):
                out[i_r, 0] = func_directional_ccd(
                    points_a, points_b, displacements_a, displacements_b, eta, thickness, max_iters
                )
            else:
                out[i_r, 0] = func_additive_ccd(
                    points_a, points_b, displacements_a, displacements_b, eta, thickness, max_iters, parallel_tolerance
                )

    return k_ccd


def make_ccd_screw_kernel(kind, is_interval):
    n_a, n_b = STENCIL_VERTS[kind]
    n = n_a + n_b

    @qd.kernel
    def k_screw(rows: qd.types.ndarray(), out: qd.types.ndarray(), max_iters_whole: int):
        for i_r in range(rows.shape[0]):
            points_a, points_b = points(rows, i_r, 0, n_a), points(rows, i_r, 3 * n_a, n_b)
            chords_a, chords_b = points(rows, i_r, 3 * n, n_a), points(rows, i_r, 3 * n + 3 * n_a, n_b)
            motion = points(rows, i_r, 6 * n, 6)
            eta, thickness = rows[i_r, 6 * n + 18], rows[i_r, 6 * n + 19]
            is_same_rigid_motion = rows[i_r, 6 * n + 20] != 0.0
            t_start, t_end = gs.qd_float(0.0), gs.qd_float(1.0)
            distance_start = gs.qd_float(0.0)
            max_iters = max_iters_whole
            if qd.static(is_interval):
                t_start, t_end, distance_start = rows[i_r, 6 * n + 21], rows[i_r, 6 * n + 22], rows[i_r, 6 * n + 23]
                max_iters = qd.cast(rows[i_r, 6 * n + 24], qd.i32)
            else:
                distance_start, _, _ = func_closest_points(points_a, points_b)
            t, n_iters, status = func_screw_ccd(
                points_a,
                chords_a,
                motion[0, :],
                motion[1, :],
                motion[2, :],
                points_b,
                chords_b,
                motion[3, :],
                motion[4, :],
                motion[5, :],
                t_start,
                t_end,
                distance_start,
                eta,
                thickness,
                max_iters,
                is_same_rigid_motion,
            )
            out[i_r, 0] = t
            if qd.static(is_interval):
                out[i_r, 1] = n_iters
                out[i_r, 2] = status

    return k_screw


@qd.kernel
def k_ccd_ph(rows: qd.types.ndarray(), out: qd.types.ndarray()):
    for i_r in range(rows.shape[0]):
        toi, is_active = func_point_halfplane_ccd(
            v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9), rows[i_r, 12], rows[i_r, 13]
        )
        out[i_r, 0] = toi
        out[i_r, 1] = qd.select(is_active, 1.0, 0.0)


@qd.kernel
def k_ccd_ph_screw(rows: qd.types.ndarray(), out: qd.types.ndarray(), max_iters: int):
    for i_r in range(rows.shape[0]):
        out[i_r, 0] = func_point_halfplane_screw_ccd(
            v3(rows, i_r, 0),
            v3(rows, i_r, 3),
            v3(rows, i_r, 6),
            v3(rows, i_r, 9),
            v3(rows, i_r, 12),
            v3(rows, i_r, 15),
            v3(rows, i_r, 18),
            rows[i_r, 21],
            rows[i_r, 22],
            max_iters,
        )


def make_candidate_kernel(kind):
    n_a, n_b = STENCIL_VERTS[kind]
    n = n_a + n_b

    @qd.kernel
    def k_candidate(
        rows: qd.types.ndarray(), out: qd.types.ndarray(), reach_scale: float, parallel_tolerance: float
    ):
        for i_r in range(rows.shape[0]):
            d_hat = rows[i_r, 9 * n]
            _, thickness, _ = func_pair_barrier_parameters(
                d_hat, d_hat, scalars(rows, i_r, 8 * n, n_a), scalars(rows, i_r, 8 * n + n_a, n_b), 1.0, 1.0
            )
            out[i_r, 0] = func_candidate_reach(
                points(rows, i_r, 0, n_a),
                points(rows, i_r, 3 * n_a, n_b),
                points(rows, i_r, 3 * n, n_a),
                points(rows, i_r, 3 * n + 3 * n_a, n_b),
                scalars(rows, i_r, 6 * n, n_a),
                scalars(rows, i_r, 6 * n + n_a, n_b),
                scalars(rows, i_r, 7 * n, n_a),
                scalars(rows, i_r, 7 * n + n_a, n_b),
                reach_scale * d_hat + thickness,
                parallel_tolerance,
                rows[i_r, 9 * n + 1] != 0.0,
            )

    return k_candidate


@qd.kernel
def k_candidate_ph(rows: qd.types.ndarray(), out: qd.types.ndarray(), reach_scale: float):
    for i_r in range(rows.shape[0]):
        is_kept = func_point_halfplane_candidate_reach(
            v3(rows, i_r, 0),
            v3(rows, i_r, 3),
            v3(rows, i_r, 6),
            v3(rows, i_r, 9),
            rows[i_r, 12],
            reach_scale * rows[i_r, 13] + rows[i_r, 14],
        )
        out[i_r, 0] = qd.select(is_kept, 1.0, 0.0)


@qd.kernel
def k_intersect(rows: qd.types.ndarray(), out: qd.types.ndarray(), tolerance: float):
    for i_r in range(rows.shape[0]):
        is_hit = func_edge_triangle_intersect(
            v3(rows, i_r, 0), v3(rows, i_r, 3), v3(rows, i_r, 6), v3(rows, i_r, 9), v3(rows, i_r, 12), tolerance
        )
        out[i_r, 0] = qd.select(is_hit, 1.0, 0.0)


def build_runners():
    """Map every case to a function running its kernel on (rows, out)."""
    runners = {
        "pt_distance": k_pt_distance,
        "ee_distance": lambda r, o: k_ee_distance(r, o, EE_PARALLEL_TOLERANCE),
        "pe_distance": k_pe_distance,
        "ee_distance_type": lambda r, o: k_ee_distance_type(r, o, EE_TYPE_PARALLEL_TOLERANCE),
        "closest_pt": k_closest_pt,
        "closest_ee": k_closest_ee,
        "closest_pe": k_closest_pe,
        "barrier": k_barrier,
        "mollified_barrier": k_mollified_barrier,
        "mollifier": lambda r, o: k_mollifier(r, o, MOLLIFIER_RATIO),
        "barrier_pt": lambda r, o: k_barrier_pt(r, o, HESSIAN_CLAMP_RATIO),
        "barrier_ee": lambda r, o: k_barrier_ee(r, o, EE_PARALLEL_TOLERANCE, MOLLIFIER_RATIO, HESSIAN_CLAMP_RATIO),
        "barrier_ee_unprefixed": lambda r, o: k_barrier_ee(
            r, o, EE_PARALLEL_TOLERANCE, MOLLIFIER_RATIO, HESSIAN_CLAMP_RATIO
        ),
        "barrier_pe": lambda r, o: k_barrier_pe(r, o, HESSIAN_CLAMP_RATIO),
        "barrier_pp": lambda r, o: k_barrier_pp(r, o, HESSIAN_CLAMP_RATIO),
        "barrier_ph": k_barrier_ph,
        "friction_ph": k_friction_ph,
        "ccd_ph": k_ccd_ph,
        "ccd_ph_screw": lambda r, o: k_ccd_ph_screw(r, o, CCD_MAX_ITERS),
        "intersect": lambda r, o: k_intersect(r, o, INTERSECTION_TOLERANCE),
        "candidate_ph": lambda r, o: k_candidate_ph(r, o, CANDIDATE_REACH_SCALE),
    }
    for kind, (n_a, n_b) in STENCIL_VERTS.items():
        runners[f"pair_{kind}"] = make_pair_kernel(n_a, n_b)
        runners[f"friction_{kind}"] = make_friction_kernel(kind)
        runners[f"frame_{kind}"] = make_frame_kernel(kind)
        additive = make_ccd_linear_kernel(kind, is_directional=False)
        directional = make_ccd_linear_kernel(kind, is_directional=True)
        screw = make_ccd_screw_kernel(kind, is_interval=False)
        candidate = make_candidate_kernel(kind)
        runners[f"ccd_additive_{kind}"] = lambda r, o, k=additive: k(r, o, CCD_MAX_ITERS, EE_TYPE_PARALLEL_TOLERANCE)
        runners[f"ccd_directional_{kind}"] = lambda r, o, k=directional: k(
            r, o, CCD_MAX_ITERS, EE_TYPE_PARALLEL_TOLERANCE
        )
        runners[f"ccd_screw_{kind}"] = lambda r, o, k=screw: k(r, o, CCD_MAX_ITERS)
        runners[f"candidate_{kind}"] = lambda r, o, k=candidate: k(
            r, o, CANDIDATE_REACH_SCALE, EE_PARALLEL_TOLERANCE
        )
    runners["frame_ph"] = make_frame_kernel("ph")
    for kind in ("pt", "ee"):
        interval = make_ccd_screw_kernel(kind, is_interval=True)
        runners[f"ccd_screw_interval_{kind}"] = lambda r, o, k=interval: k(r, o, CCD_MAX_ITERS)
    missing = set(CASES) - set(runners)
    assert not missing, missing
    return runners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--fast-math", choices=("true", "false"), required=True)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--sum-order", choices=("quadrants", "eigen"), default="quadrants")
    args = parser.parse_args()

    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", logging_level="warning")
    qd.lang.impl.current_cfg().fast_math = args.fast_math == "true"
    if args.sum_order == "eigen":
        matrix_ops._reduce = _reduce_eigen_order
    print(
        f"[genesis] {gs.__file__}, backend {args.backend}, fast_math {qd.lang.impl.current_cfg().fast_math}, "
        f"sum order {args.sum_order}"
    )

    runners = build_runners()
    outputs = {}
    with np.load(HERE / "data" / "inputs.npz") as inputs:
        for key in inputs.files:
            name = key.split("/")[0]
            if args.cases and name not in args.cases:
                continue
            rows = torch.as_tensor(inputs[key], dtype=gs.tc_float, device=gs.device).contiguous()
            out = torch.full((rows.shape[0], n_outputs(CASES[name])), torch.nan, dtype=gs.tc_float, device=gs.device)
            start = time.perf_counter()
            runners[name](rows, out)
            outputs[key] = out.cpu().numpy()
            print(f"[genesis] {key}: {rows.shape[0]} rows, {time.perf_counter() - start:.1f} s", flush=True)
    suffix = "" if args.sum_order == "quadrants" else "_sum-order-eigen"
    path = HERE / "data" / f"out_genesis_{args.backend}_fast-math-{args.fast_math}{suffix}.npz"
    np.savez(path, **outputs)
    print(f"[genesis] wrote {path}")


if __name__ == "__main__":
    main()
