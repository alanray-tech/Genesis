"""Probes of the contact of a Genesis scene for the parity benches: the start point of the next step published into the
contact surface, and the active pairs of the first Newton iteration there, in contact vertex indices."""

import numpy as np
import torch

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.fem_solver import FEMState
from genesis.engine.solvers.ipc_contact import (
    IPCInfo,
    IPCPairs,
    IPCState,
    func_flagged_distance2,
    func_pair_parameters,
    func_stencil_positions,
)
from genesis.engine.solvers.newton_solver import kernel_publish_contact
from genesis.engine.solvers.rigid.contact_proxy import kernel_proxies_init_pose
from genesis.utils.misc import qd_to_numpy


@qd.kernel
def kernel_set_start_point(fem_state: FEMState):
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape):
        fem_state.vertices.pos_next[i_v, i_b] = fem_state.vertices.pos[i_v, i_b]


@qd.func
def func_classify(i_kind: qd.template(), ipc_pairs: IPCPairs, ipc_state: IPCState, ipc_info: IPCInfo):
    for i_p in range(qd.min(ipc_pairs.counts[i_kind], ipc_pairs.pt.shape[0])):
        pair = qd.Vector.zero(gs.qd_int, 3)
        if qd.static(i_kind == 1):
            pair = ipc_pairs.ee[i_p]
        else:
            pair = ipc_pairs.pt[i_p]
        verts_idx, d_hat, thickness, _, _, _ = func_pair_parameters(pair[1], pair[2], ipc_info, i_kind == 1)
        x = func_stencil_positions(pair[0], verts_idx, ipc_state.vertices.pos)
        flag, distance2, _ = func_flagged_distance2(x, ipc_info, i_kind == 1)
        is_active = distance2 > thickness * thickness and distance2 < (d_hat + thickness) * (d_hat + thickness)
        if qd.static(i_kind == 1):
            ipc_pairs.ee_flag[i_p] = qd.select(is_active, flag, 0)
        else:
            ipc_pairs.pt_flag[i_p] = qd.select(is_active, flag, 0)


@qd.kernel
def kernel_classify(ipc_pairs: IPCPairs, ipc_state: IPCState, ipc_info: IPCInfo):
    """The distance flags of the candidates in the band at the start point, 0 outside, as 'func_contact_assemble'."""
    func_classify(0, ipc_pairs, ipc_state, ipc_info)
    func_classify(1, ipc_pairs, ipc_state, ipc_info)


def init_proxies(scene) -> None:
    """Set every rigid contact proxy at the inertial frame of its link, as the first step does."""
    newton, rigid = scene.sim.newton_solver, scene.sim.rigid_solver
    if newton.newton_config.has_rigid_vertices:
        kernel_proxies_init_pose(
            torch.ones(scene.sim._B, dtype=gs.tc_int, device=gs.device),
            newton.data_manager.proxies_state,
            rigid.dyn_state,
            newton.data_manager.proxies_info,
            rigid.dyn_info,
            rigid.rigid_config,
        )


def publish_start_point(scene) -> None:
    """Set the start point of the next step at the current positions and publish it into the contact surface."""
    newton, fem, rigid = scene.sim.newton_solver, scene.sim.fem_solver, scene.sim.rigid_solver
    kernel_set_start_point(fem.fem_state)
    kernel_publish_contact(
        fem.n_vertices,
        newton.ipc_state,
        fem.fem_state,
        newton.data_manager.proxies_state,
        rigid.dyn_state,
        newton.data_manager.proxies_info,
        newton.data_manager.proxy_vertices_info,
        rigid.dyn_info,
        rigid.rigid_config,
        newton.newton_config,
    )


def canonical(stencil: np.ndarray, flag: np.ndarray, is_edge_edge: bool) -> dict:
    """Map every pair to its canonical stencil, with its distance flag permuted alike: the vertices of a triangle and of
    an edge increasing, the edges of an edge-edge pair increasing."""
    pairs = {}
    for row, f in zip(stencil.tolist(), flag.tolist()):
        bits = [(f >> k) & 1 for k in range(4)]
        if is_edge_edge:
            edges = [sorted(zip(row[0:2], bits[0:2])), sorted(zip(row[2:4], bits[2:4]))]
            edges.sort(key=lambda edge: [v for v, _ in edge])
            items = edges[0] + edges[1]
        else:
            items = [(row[0], bits[0])] + sorted(zip(row[1:4], bits[1:4]))
        key = tuple(v for v, _ in items)
        pairs[key] = sum(b << k for k, (_, b) in enumerate(items))
    return pairs


def compare_pairs(genesis_pairs, cgq_stencil, cgq_flag, is_edge_edge) -> dict:
    """Count the pairs of each side, those of one side alone, and the common pairs whose distance flags differ."""
    genesis_keys = canonical(*genesis_pairs, is_edge_edge)
    cgq_keys = canonical(cgq_stencil, cgq_flag, is_edge_edge)
    common = genesis_keys.keys() & cgq_keys.keys()
    return {
        "n_genesis": len(genesis_keys),
        "n_cgq": len(cgq_keys),
        "only_genesis": len(genesis_keys.keys() - cgq_keys.keys()),
        "only_cgq": len(cgq_keys.keys() - genesis_keys.keys()),
        "flag_mismatch": sum(genesis_keys[key] != cgq_keys[key] for key in common),
    }


def genesis_active_pairs(scene) -> dict:
    """The active pairs of the first Newton iteration of the next step, from the state of the scene, as (stencils, flags)
    per kind in contact vertex indices."""
    newton = scene.sim.newton_solver
    contact = newton.ipc_contact
    publish_start_point(scene)
    contact.search(newton.ipc_state, newton.ipc_info, newton.newton_config, is_swept=False)
    kernel_classify(contact.pairs, newton.ipc_state, newton.ipc_info)
    tris = qd_to_numpy(newton.ipc_info.triangles.verts_idx)
    edges = qd_to_numpy(newton.ipc_info.edges.verts_idx)
    n_pt, n_ee = contact.n_pairs
    pt, pt_flag = qd_to_numpy(contact.pairs.pt)[:n_pt], qd_to_numpy(contact.pairs.pt_flag)[:n_pt]
    ee, ee_flag = qd_to_numpy(contact.pairs.ee)[:n_ee], qd_to_numpy(contact.pairs.ee_flag)[:n_ee]
    pt_stencil = np.column_stack([pt[:, 1], tris[pt[:, 2]]])
    ee_stencil = np.column_stack([edges[ee[:, 1]], edges[ee[:, 2]]])
    return {
        "pt": (pt_stencil[pt_flag != 0], pt_flag[pt_flag != 0]),
        "ee": (ee_stencil[ee_flag != 0], ee_flag[ee_flag != 0]),
    }
