"""Conformance of the reduced-KKT Newton systems Genesis solves with CGQ aeb58956's dense reference
(tests/rigid_proxy_kkt_reference.py), on the states Genesis meets in its own run of box_on_cloth (rigid_scenes.py, a
free box dropped onto a cloth lying on the ground, double precision).

At every Newton iteration in contact, once the preconditioned conjugate gradient (PCG) ends, the system of Genesis is
captured: the Hessian of the rows (the cloth and the proxy) from its BCOO entries, the mass matrix of the dofs, the
gradients, the reduced right-hand side b, the solution x and the residual r. The Jacobian J of the target pose of the
proxy (the inertial frame of the box at q1(x) = q0 (+) (dt v0 + x)) is taken by central differences on the host, and the
reference builds c, A, B and the reduced system from it. Checked:
- the maps of Genesis against the reference: the tangent map Jl Jr^-1, the normal map Jl and A d_p = -c (C P = 0 and
  C d_p = -c),
- H d_p of the proxy rows against the dense product,
- b of the dofs against the reference P^T (g + H d_p),
- the reduced product: H_red x against b - r, the PCG invariant (the reduced matvec of Genesis on the solution),
- the solution against the dense solve of the reduced system, beside the relative residual of the PCG,
- the FK-defect certificate: the exact surface mismatch h along the whole first step of the line search, sampled densely
  on the host, against the path limit of the proxy.
The constraint is compared in absolute terms (it vanishes at most iterates), C d_p = -c against 1 + |c|, H d_p against
|H_pp| |d_p|, b of the dofs against its two terms (they cancel near convergence), and the rest against the largest
entry of the reference quantity. The solution differs from the dense one by what the PCG tolerance leaves.

    python bench/ipc_rigid/kkt_conformance.py --backend gpu|cpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.numerics.newton_rows import NewtonSegment
from genesis.utils.misc import qd_to_numpy

sys.path.insert(0, str(Path(__file__).parent))

import rigid_scenes as scenes  # noqa: E402

CGQ_TESTS = Path(r"C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-p2a-aeb58956\tests")
sys.path.insert(0, str(CGQ_TESTS))
import rigid_proxy_kkt_reference as ref  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--fd-step", type=float, default=1e-6)
parser.add_argument("--n-samples", type=int, default=400)
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")

import genesis.engine.solvers.newton_solver as newton_solver  # noqa: E402

rigid_scene = next(s for s in scenes.make_scenes(16) if s.name == "box_on_cloth")
(cloth_spec,) = rigid_scene.cloths
(box,) = rigid_scene.boxes
mesh_path = Path(args.data) / "kkt_conformance_cloth.obj"
mesh_path.write_text(
    trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth_spec.verts, cloth_spec.tris, process=False), digits=17)
)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
    newton_options=gs.options.NewtonOptions(velocity_tol=scenes.VELOCITY_TOL),
    ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
    show_viewer=False,
)
scene.add_entity(
    morph=gs.morphs.Mesh(file=str(mesh_path)),
    material=gs.materials.FEM.Cloth(
        E=scenes.YOUNGS_MODULUS,
        rho=scenes.MASS_DENSITY,
        thickness=scenes.THICKNESS,
        shear_modulus=scenes.SHEAR_MODULUS,
        bending=rigid_scene.bending,
        bending_E=scenes.BENDING_YOUNGS_MODULUS,
    ),
)
scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
box_entity = scene.add_entity(
    morph=gs.morphs.Box(size=box.size, pos=box.pos, quat=box.quat),
    material=gs.materials.Rigid(rho=box.density, coup_friction=gs.materials.FEM.Cloth().friction_mu),
)
scene.build()
newton, rigid = scene.sim.newton_solver, scene.sim.rigid_solver
data_manager = newton.data_manager
proxies, proxies_state, proxies_envs = data_manager.proxies, data_manager.proxies_state, data_manager.proxies_envs
rigid_dofs = data_manager.rigid_dofs
dt = scenes.DT


def rel(value, reference) -> float:
    return float(np.abs(value - reference).max() / (np.abs(reference).max() + 1e-300))


def target_pose(x, q0_t, R0, v0, inertial_pos, inertial_R):
    """The pose of the inertial frame of the box at q1(x), the free joint taking x in the world for its translation and
    in the body frame for its rotation."""
    t = q0_t + dt * v0[:3] + x[:3]
    R = R0 @ ref.so3_exp(dt * v0[3:] + x[3:])
    return t + R @ inertial_pos, R @ inertial_R


rows = {key: [] for key in ("tangent", "normal", "C_dp", "H_dp", "b_dofs", "matvec", "dense_solution", "pcg_residual")}
rows.update(fk_margin=[], constraint=[], constraint_norm=[])
captured = {}
start_line_search = newton_solver.kernel_start_line_search


def captured_start_line_search(newton_rows, newton_envs, newton_loops, *more):
    is_capturing = bool(qd_to_numpy(newton_envs.is_active)[0]) and bool(
        np.abs(qd_to_numpy(proxies.gradient)[0, 0]).max() > 0.0
    )
    if is_capturing:
        capture(newton_rows)
    start_line_search(newton_rows, newton_envs, newton_loops, *more)
    if is_capturing:
        check_certificate(newton_envs)


def capture(newton_rows) -> None:
    segments_start = qd_to_numpy(newton_rows.segments_start)
    n_rows = int(segments_start[-1])
    proxy_row = int(segments_start[int(NewtonSegment.PROXIES)])
    n_entries = int(qd_to_numpy(newton_rows.matrix.n_entries))
    entries_row = qd_to_numpy(newton_rows.matrix.entries_row)[:n_entries]
    entries_col = qd_to_numpy(newton_rows.matrix.entries_col)[:n_entries]
    entries_value = qd_to_numpy(newton_rows.matrix.entries_value)[: 9 * n_entries].reshape(n_entries, 3, 3)
    H_rows = np.zeros((3 * n_rows, 3 * n_rows))
    for i_r, j_r, block in zip(entries_row, entries_col, entries_value):
        H_rows[3 * i_r : 3 * i_r + 3, 3 * j_r : 3 * j_r + 3] += block
        if i_r != j_r:
            H_rows[3 * j_r : 3 * j_r + 3, 3 * i_r : 3 * i_r + 3] += block.T
    # The rows of the cloth, then the translation and rotation rows of the proxy, the last rows of the system
    assert proxy_row + 2 == n_rows
    n_other = 3 * proxy_row
    b_rows = qd_to_numpy(newton_rows.gradient)[:n_rows].reshape(-1)
    x_rows = qd_to_numpy(newton_rows.solution)[:n_rows].reshape(-1)
    r_rows = qd_to_numpy(newton_rows.r)[:n_rows].reshape(-1)
    b_dofs = qd_to_numpy(rigid_dofs.gradient)[:, 0]
    x_dofs = qd_to_numpy(rigid_dofs.solution)[:, 0]
    r_dofs = qd_to_numpy(rigid_dofs.r)[:, 0]
    x_iterate = qd_to_numpy(rigid_dofs.x)[:, 0]
    Ma = qd_to_numpy(rigid_dofs.Ma)[:, 0]
    qf_smooth = qd_to_numpy(rigid.dyn_state.dofs.qf_smooth)[:, 0]
    M = rigid.get_mass_mat().cpu().numpy()
    if not np.allclose(M, M.T):
        M = np.tril(M) + np.tril(M, -1).T
    qpos = box_entity.get_qpos().cpu().numpy()
    v0 = box_entity.get_dofs_velocity().cpu().numpy()
    inertial_pos = qd_to_numpy(proxies_state.inertial_pos)[0, 0]
    inertial_R = gu.quat_to_R(qd_to_numpy(proxies_state.inertial_quat)[0, 0])
    R0 = gu.quat_to_R(qpos[3:7])
    # The pose of the proxy at the iterate, which the line search copies to its start point once it starts
    t_p = qd_to_numpy(proxies_state.pos)[0, 0]
    R_p = gu.quat_to_R(qd_to_numpy(proxies_state.quat)[0, 0])
    t_target, R_target = target_pose(x_iterate, qpos[:3], R0, v0, inertial_pos, inertial_R)
    J = np.zeros((6, 6))
    for k in range(6):
        step = np.zeros(6)
        step[k] = args.fd_step
        t_plus, R_plus = target_pose(x_iterate + step, qpos[:3], R0, v0, inertial_pos, inertial_R)
        t_minus, R_minus = target_pose(x_iterate - step, qpos[:3], R0, v0, inertial_pos, inertial_R)
        J[:3, k] = (t_plus - t_minus) / (2.0 * args.fd_step)
        J[3:, k] = ref.so3_log(R_plus @ R_minus.T) / (2.0 * args.fd_step)
    c, A, B = ref.pose_constraint_jacobians(t_p, R_p, t_target, R_target, J)
    A_inv = np.linalg.inv(A)
    c_genesis = qd_to_numpy(proxies.constraint)[0, 0]
    rows["constraint"].append(float(np.abs(c_genesis - c).max()))
    rows["constraint_norm"].append(float(np.abs(c).max()))

    tangent = qd_to_numpy(proxies.tangent_rotation)[0, 0]
    normal = qd_to_numpy(proxies.normal_rotation)[0, 0]
    particular = qd_to_numpy(proxies.particular)[0, 0]
    rows["tangent"].append(rel(tangent, A_inv[3:, 3:] @ ref.so3_right_jacobian_inverse(c[3:])))
    rows["normal"].append(rel(normal, A_inv[3:, 3:]))
    rows["C_dp"].append(float(np.abs(A @ particular + c).max() / (1.0 + np.abs(c).max())))

    n_full = n_other + 6 + 6
    H_full = np.zeros((n_full, n_full))
    H_full[: n_other + 6, : n_other + 6] = H_rows
    H_full[n_other + 6 :, n_other + 6 :] = M
    d_p = np.zeros(n_full)
    d_p[n_other : n_other + 6] = particular
    H_dp = H_full @ d_p
    H_dp_scale = np.abs(H_full[n_other : n_other + 6, n_other : n_other + 6]).max() * np.abs(particular).max()
    H_dp_error = np.abs(qd_to_numpy(proxies.hessian_particular)[0, 0] - H_dp[n_other : n_other + 6]).max()
    rows["H_dp"].append(float(H_dp_error / (H_dp_scale + 1e-300)) if H_dp_scale > 0.0 else 0.0)
    g_full = np.concatenate(
        [b_rows[:n_other] - H_dp[:n_other], qd_to_numpy(proxies.gradient)[0, 0], dt * dt * (Ma - qf_smooth)]
    )
    reduced = ref.reduce_proxy_kkt(H_full, g_full, A, B, c, n_other)
    # The reduced gradient of the dofs is the sum of the rigid gradient and the restricted wrench of the proxy, which
    # cancel near convergence: the error is measured against the size of these terms
    restricted = reduced.gradient[n_other:] - g_full[n_other + 6 :]
    b_scale = max(np.abs(g_full[n_other + 6 :]).max(), np.abs(restricted).max())
    rows["b_dofs"].append(float(np.abs(b_dofs - reduced.gradient[n_other:]).max() / (b_scale + 1e-300)))
    x_reduced = np.concatenate([x_rows[:n_other], x_dofs])
    b_reduced = np.concatenate([b_rows[:n_other], b_dofs])
    r_reduced = np.concatenate([r_rows[:n_other], r_dofs])
    rows["matvec"].append(rel(reduced.hessian @ x_reduced, b_reduced - r_reduced))
    x_dense = np.linalg.solve(reduced.hessian, reduced.gradient)
    rows["dense_solution"].append(rel(x_reduced, x_dense))
    rows["pcg_residual"].append(
        float(np.linalg.norm(reduced.hessian @ x_reduced - reduced.gradient) / np.linalg.norm(reduced.gradient))
    )
    captured.update(qpos=qpos, v0=v0, inertial_pos=inertial_pos, inertial_R=inertial_R)


def check_certificate(newton_envs) -> None:
    """Sample the exact surface mismatch along the first step of the line search, to the step it starts at."""
    alpha_max = float(qd_to_numpy(newton_envs.alpha)[0])
    x_start = qd_to_numpy(rigid_dofs.x_start)[:, 0]
    dx = qd_to_numpy(rigid_dofs.dx)[:, 0]
    t_iter = qd_to_numpy(proxies.pos_iter)[0, 0]
    R_iter = gu.quat_to_R(qd_to_numpy(proxies.quat_iter)[0, 0])
    direction_pos = qd_to_numpy(proxies.direction_pos)[0, 0]
    direction_rot = qd_to_numpy(proxies.direction_rot)[0, 0]
    path_limit = float(qd_to_numpy(proxies.path_limit)[0, 0])
    radius = float(qd_to_numpy(proxies_state.surface_radius)[0, 0])
    R0 = gu.quat_to_R(captured["qpos"][3:7])
    margin = -np.inf
    for alpha in np.linspace(0.0, alpha_max, args.n_samples):
        t_target, R_target = target_pose(
            x_start + alpha * dx,
            captured["qpos"][:3],
            R0,
            captured["v0"],
            captured["inertial_pos"],
            captured["inertial_R"],
        )
        c = ref.pose_constraint(
            t_iter + alpha * direction_pos, ref.so3_exp(alpha * direction_rot) @ R_iter, t_target, R_target
        )
        margin = max(margin, (np.linalg.norm(c[:3]) + radius * np.linalg.norm(c[3:]) - path_limit) / (1.0 + path_limit))
    rows["fk_margin"].append(float(margin))


newton_solver.kernel_start_line_search = captured_start_line_search
for _ in range(rigid_scene.n_frames):
    scene.step()

summary = {"backend": args.backend, "n_captures": len(rows["tangent"])}
for key, values in rows.items():
    values = np.array(values)
    summary[key] = {"max": float(values.max()), "median": float(np.median(values))}
    print(f"{key:16s} max {values.max():.3e} median {np.median(values):.3e}", flush=True)
out = Path(args.results) / f"kkt_conformance_{args.backend}.json"
out.write_text(json.dumps(summary, indent=1))
print(f"{summary['n_captures']} captures -> {out}")
