"""Replay every frame of the CGQ cloth runs in Genesis, and run the whole trajectory, in double precision.

Replay: the state of CGQ at frame k (positions and velocities) is loaded into Genesis, which steps once, and its
positions are compared with CGQ's frame k + 1. The bound is 2 * velocity_tol * dt, both solves stopping within
velocity_tol * dt of the solution. Trajectory: Genesis runs from frame 0 on its own, compared with CGQ frame by frame.

The scenes of the contact between cloths also compare, on every replayed frame, the active pairs of the first Newton
iteration: the candidates Genesis finds at the start point of the step, classified as its first assembly does, against
those CGQ dumped (see cgq_side.py), by their global vertex stencils, and count the pairs whose distance flags differ.
Every frame of the Genesis trajectory is checked for an edge crossing a triangle.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.fem_solver import FEMInfo, FEMState
from genesis.engine.solvers.ipc_contact import IPCInfo, IPCPairs, func_flagged_distance2, func_pair_parameters, func_stencil_positions
from genesis.utils.misc import qd_to_numpy

sys.path.insert(0, str(Path(__file__).parent))

import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--resolution", type=int, default=10)
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--fast-math", type=int, default=1)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
parser.add_argument("--max-iterations", type=int, default=1024)
parser.add_argument("--reference-tag", default="", help="suffix of the CGQ runs (see cgq_side.py)")
parser.add_argument("--tag", default="")
parser.add_argument(
    "--replay-frames", type=int, nargs="*", default=None, help="replay these frames alone, without the trajectory"
)
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
data_dir, results_dir = Path(args.data), Path(args.results)
results_dir.mkdir(parents=True, exist_ok=True)

# Genesis compiles with fast_math outside debug mode alone
gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning", debug=not args.fast_math)
bound = 2.0 * args.velocity_tol * scenes.DT
summary = {
    "backend": args.backend,
    "fast_math": bool(args.fast_math),
    "velocity_tol": args.velocity_tol,
    "bound": bound,
    "scenes": {},
}


@qd.kernel
def kernel_set_start_point(fem_state: FEMState):
    for i_v, i_b in qd.ndrange(*fem_state.vertices.pos.shape):
        fem_state.vertices.pos_next[i_v, i_b] = fem_state.vertices.pos[i_v, i_b]


@qd.func
def func_classify(i_kind: qd.template(), ipc_pairs: IPCPairs, fem_state: FEMState, fem_info: FEMInfo, ipc_info: IPCInfo):
    for i_p in range(qd.min(ipc_pairs.counts[i_kind], ipc_pairs.pt.shape[0])):
        pair = qd.Vector.zero(gs.qd_int, 3)
        if qd.static(i_kind == 1):
            pair = ipc_pairs.ee[i_p]
        else:
            pair = ipc_pairs.pt[i_p]
        verts_idx, d_hat, thickness, _, _, _ = func_pair_parameters(pair[1], pair[2], fem_info, i_kind == 1)
        x = func_stencil_positions(pair[0], verts_idx, fem_state.vertices.pos_next)
        flag, distance2, _ = func_flagged_distance2(x, ipc_info, i_kind == 1)
        is_active = distance2 > thickness * thickness and distance2 < (d_hat + thickness) * (d_hat + thickness)
        if qd.static(i_kind == 1):
            ipc_pairs.ee_flag[i_p] = qd.select(is_active, flag, 0)
        else:
            ipc_pairs.pt_flag[i_p] = qd.select(is_active, flag, 0)


@qd.kernel
def kernel_classify(ipc_pairs: IPCPairs, fem_state: FEMState, fem_info: FEMInfo, ipc_info: IPCInfo):
    """The distance flags of the candidates in the band at the start point, 0 outside, as 'func_contact_assemble'."""
    func_classify(0, ipc_pairs, fem_state, fem_info, ipc_info)
    func_classify(1, ipc_pairs, fem_state, fem_info, ipc_info)


def genesis_active_pairs(scene) -> dict:
    """The active pairs of the first Newton iteration of the next step, from the state of the scene."""
    newton, fem = scene.sim.newton_solver, scene.sim.fem_solver
    contact = newton.ipc_contact
    kernel_set_start_point(fem.fem_state)
    contact.search(fem.fem_state, newton.ipc_state, fem.fem_info, newton.ipc_info, newton.newton_config, is_swept=False)
    kernel_classify(contact.pairs, fem.fem_state, fem.fem_info, newton.ipc_info)
    tris = qd_to_numpy(fem.fem_info.cloth_triangles.verts_idx)
    edges = qd_to_numpy(fem.fem_info.cloth_edges.verts_idx)
    n_pt, n_ee = contact.n_pairs
    pt, pt_flag = qd_to_numpy(contact.pairs.pt)[:n_pt], qd_to_numpy(contact.pairs.pt_flag)[:n_pt]
    ee, ee_flag = qd_to_numpy(contact.pairs.ee)[:n_ee], qd_to_numpy(contact.pairs.ee_flag)[:n_ee]
    pt_stencil = np.column_stack([pt[:, 1], tris[pt[:, 2]]])
    ee_stencil = np.column_stack([edges[ee[:, 1]], edges[ee[:, 2]]])
    return {
        "pt": (pt_stencil[pt_flag != 0], pt_flag[pt_flag != 0]),
        "ee": (ee_stencil[ee_flag != 0], ee_flag[ee_flag != 0]),
    }


def compare_pairs(genesis_pairs, cgq_stencil, cgq_flag) -> dict:
    """The pairs of each side alone, by stencil, and the common pairs whose distance flags differ."""
    stencil, flag = genesis_pairs
    genesis_keys = {tuple(row): f for row, f in zip(stencil.tolist(), flag.tolist())}
    cgq_keys = {tuple(row): f for row, f in zip(cgq_stencil.tolist(), cgq_flag.tolist())}
    common = genesis_keys.keys() & cgq_keys.keys()
    return {
        "n_genesis": len(genesis_keys),
        "n_cgq": len(cgq_keys),
        "only_genesis": len(genesis_keys.keys() - cgq_keys.keys()),
        "only_cgq": len(cgq_keys.keys() - genesis_keys.keys()),
        "flag_mismatch": sum(genesis_keys[key] != cgq_keys[key] for key in common),
    }


contact_scenes = scenes.make_contact_scenes(args.resolution)
contact_names = {cloth_scene.name for cloth_scene in contact_scenes}
for cloth_scene in scenes.make_scenes(args.n) + contact_scenes:
    if args.only and cloth_scene.name not in args.only:
        continue
    is_contact = cloth_scene.name in contact_names
    reference = np.load(data_dir / f"cgq_{cloth_scene.name}{args.reference_tag}.npz")
    cgq_positions, cgq_velocities = reference["positions"], reference["velocities"]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
        newton_options=gs.options.NewtonOptions(
            velocity_tol=args.velocity_tol, max_iterations=args.max_iterations, line_search_max_iterations=12
        ),
        ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
        show_viewer=False,
    )
    if cloth_scene.has_ground:
        scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
    entities, slices, v_start = [], [], 0
    for i_cloth, cloth in enumerate(cloth_scene.cloths):
        mesh_path = data_dir / f"{cloth_scene.name}_{i_cloth}.obj"
        mesh_path.write_text(
            trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth.verts, cloth.tris, process=False), digits=17)
        )
        entities.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(file=str(mesh_path)),
                material=gs.materials.FEM.Cloth(
                    E=scenes.YOUNGS_MODULUS,
                    rho=scenes.MASS_DENSITY,
                    thickness=scenes.THICKNESS,
                    shear_modulus=scenes.SHEAR_MODULUS,
                    bending=cloth_scene.bending,
                    bending_E=scenes.BENDING_YOUNGS_MODULUS,
                ),
            )
        )
        slices.append(slice(v_start, v_start + len(cloth.verts)))
        v_start += len(cloth.verts)
    scene.build()

    def get_positions():
        return np.concatenate([entity.get_verts_pos().cpu().numpy() for entity in entities])

    def pin():
        for entity, cloth in zip(entities, cloth_scene.cloths):
            if cloth.pinned.any():
                entity.set_vertex_constraints(verts_idx_local=np.flatnonzero(cloth.pinned))

    init_error = float(np.abs(get_positions() - cgq_positions[0]).max())
    pin()

    # One-step replay of every frame of CGQ, in order, so that the adaptive stiffness of a frame comes from the step
    # before it
    replay_errors, step_ms, pairs_rows = [], [], []
    replay_frames = range(cloth_scene.n_frames) if args.replay_frames is None else args.replay_frames
    for k in replay_frames:
        for entity, part in zip(entities, slices):
            entity.set_verts_pos(cgq_positions[k][part])
            entity.set_verts_vel(cgq_velocities[k][part])
        if is_contact:
            genesis_pairs = genesis_active_pairs(scene)
            row = {}
            for ch in ("pt", "ee"):
                offsets = reference[f"{ch}_offsets"]
                row[ch] = compare_pairs(
                    genesis_pairs[ch],
                    reference[f"{ch}_stencil"][offsets[k] : offsets[k + 1]],
                    reference[f"{ch}_flag"][offsets[k] : offsets[k + 1]],
                )
            pairs_rows.append(row)
        t = time.perf_counter()
        scene.step()
        positions = get_positions()
        step_ms.append(1e3 * (time.perf_counter() - t))
        replay_errors.append(float(np.abs(positions - cgq_positions[k + 1]).max()))
    if args.replay_frames is not None:
        errors = ", ".join(f"frame {k}: {e:.3e}" for k, e in zip(replay_frames, replay_errors))
        print(f"{cloth_scene.name} [{args.backend}] tol {args.velocity_tol:g}: {errors}", flush=True)
        scene.destroy()
        continue

    # The whole trajectory, from the initial state
    scene.reset()
    pin()
    trajectory_errors, n_intersections = [], []
    newton, fem = scene.sim.newton_solver, scene.sim.fem_solver
    for k in range(cloth_scene.n_frames):
        scene.step()
        positions = get_positions()
        trajectory_errors.append(float(np.abs(positions - cgq_positions[k + 1]).max()))
        if is_contact:
            kernel_set_start_point(fem.fem_state)
            n_intersections.append(len(newton.ipc_contact.find_intersections(fem.fem_state, fem.fem_info, newton.ipc_info)))

    replay_errors = np.array(replay_errors)
    result = {
        "init_error": init_error,
        "replay_max_error": float(replay_errors.max()),
        "replay_worst_frame": int(replay_errors.argmax()),
        "replay_frames_above_bound": int((replay_errors > bound).sum()),
        "replay_errors": replay_errors.tolist(),
        "trajectory_errors": trajectory_errors,
        "trajectory_max_error": float(max(trajectory_errors)),
        "genesis_step_ms_median": float(np.median(step_ms[1:])),
        "cgq_step_ms_median": float(np.median(reference["step_ms"][1:])),
        "cgq_newton_iters_max": int(reference["newton_iters"].max()),
    }
    if cloth_scene.has_ground:
        result["genesis_rest_min_y"] = float(positions[:, 1].min())
        result["cgq_rest_min_y"] = float(cgq_positions[-1][:, 1].min())
    pairs_text = ""
    if is_contact:
        totals = {
            ch: {key: int(sum(row[ch][key] for row in pairs_rows)) for key in pairs_rows[0][ch]} for ch in ("pt", "ee")
        }
        result["pairs"] = pairs_rows
        result["pairs_totals"] = totals
        result["pairs_frames_differing"] = int(
            sum(any(row[ch]["only_genesis"] or row[ch]["only_cgq"] for ch in ("pt", "ee")) for row in pairs_rows)
        )
        result["trajectory_intersections"] = int(sum(n_intersections))
        pairs_text = (
            f", pairs {totals} ({result['pairs_frames_differing']} frames differ), "
            f"intersections {result['trajectory_intersections']}"
        )
    summary["scenes"][cloth_scene.name] = result
    print(
        f"{cloth_scene.name} [{args.backend}]: replay max {result['replay_max_error']:.3e} (bound {bound:.1e}, "
        f"{result['replay_frames_above_bound']} frames above, worst frame {result['replay_worst_frame']}), "
        f"trajectory max {result['trajectory_max_error']:.3e}, step {result['genesis_step_ms_median']:.2f} ms"
        f"{pairs_text}",
        flush=True,
    )
    scene.destroy()

out = results_dir / f"genesis_{args.backend}{args.tag}.json"
out.write_text(json.dumps(summary, indent=1))
print(f"PASS: {all(s['replay_frames_above_bound'] == 0 for s in summary['scenes'].values())} -> {out}")
