"""Genesis side of the rigid parity with CGQ aeb58956, in double precision (see scenes.py).

--dump-mesh builds every scene and saves the mesh of its rigid contact proxies in world coordinates, which cgq_side.py
loads. Otherwise, as bench/ipc_cloth/genesis_side.py: every frame of the CGQ run is replayed for one step (the state of
CGQ at frame k loaded into Genesis, its positions compared with CGQ's frame k + 1, bound 2 * velocity_tol * dt), the
active pairs of the first Newton iteration of every replayed frame compared with CGQ's, and the whole trajectory run from
frame 0, every frame of it checked for an edge crossing a triangle. Pairs are compared by their stencils in a canonical
order (the vertices of a triangle and of an edge increasing, the edges of an edge-edge pair increasing), their distance
flags permuted alike.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ipc_cloth"))

import rigid_scenes as scenes  # noqa: E402
from contact_probe import compare_pairs, genesis_active_pairs, init_proxies, publish_start_point  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--fast-math", type=int, default=1)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
parser.add_argument("--dump-mesh", action="store_true")
parser.add_argument("--tag", default="")
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
data_dir, results_dir = Path(args.data), Path(args.results)
data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning", debug=not args.fast_math)
bound = 2.0 * args.velocity_tol * scenes.DT
from newton_counters import NewtonCounters, compare_counts  # noqa: E402

counters = NewtonCounters(gs.options.NewtonOptions().pcg_max_iterations)
summary = {"backend": args.backend, "fast_math": bool(args.fast_math), "velocity_tol": args.velocity_tol, "scenes": {}}


for rigid_scene in scenes.make_scenes(args.n):
    if args.only and rigid_scene.name not in args.only:
        continue
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
        newton_options=gs.options.NewtonOptions(
            velocity_tol=args.velocity_tol, max_iterations=1024, line_search_max_iterations=12
        ),
        ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
        show_viewer=False,
    )
    entities = []
    for i_cloth, cloth in enumerate(rigid_scene.cloths):
        mesh_path = data_dir / f"{rigid_scene.name}_{i_cloth}.obj"
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
                    bending=rigid_scene.bending,
                    bending_E=scenes.BENDING_YOUNGS_MODULUS,
                ),
            )
        )
    if rigid_scene.has_ground:
        scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
    (box,) = rigid_scene.boxes
    box_entity = scene.add_entity(
        morph=gs.morphs.Box(size=box.size, pos=box.pos, quat=box.quat, fixed=box.is_fixed),
        # The friction of a rigid-cloth pair is the mean of both sides, CGQ's default model being the cloth's
        material=gs.materials.Rigid(rho=box.density, coup_friction=gs.materials.FEM.Cloth().friction_mu),
    )
    scene.build()
    newton = scene.sim.newton_solver

    def set_initial_spin() -> None:
        if not box.is_fixed:
            omega_link = scenes.quat_to_R(box.quat).T @ np.array(box.omega)
            box_entity.set_dofs_velocity(np.concatenate([np.zeros(3), omega_link]))

    set_initial_spin()
    n_cloth_verts = sum(len(cloth.verts) for cloth in rigid_scene.cloths)

    if args.dump_mesh:
        init_proxies(scene)
        publish_start_point(scene)
        verts = qd_to_numpy(newton.ipc_state.vertices.pos)[n_cloth_verts:, 0]
        tris = qd_to_numpy(newton.ipc_info.triangles.verts_idx)
        tris = tris[tris.min(axis=1) >= n_cloth_verts] - n_cloth_verts
        np.savez(data_dir / f"rigid_mesh_{rigid_scene.name}.npz", verts=verts, tris=tris)
        print(f"{rigid_scene.name}: rigid mesh {len(verts)} vertices, {len(tris)} triangles", flush=True)
        scene.destroy()
        continue

    reference = np.load(data_dir / f"cgq_{rigid_scene.name}.npz")
    cgq_positions, cgq_velocities = reference["positions"], reference["velocities"]
    # The references of a fixed box predate the rigid states, which it leaves unused
    if "rigid_states" in reference:
        cgq_rigid = reference["rigid_states"]
    else:
        cgq_rigid = np.zeros((rigid_scene.n_frames + 1, 20))
    slices, v_start = [], 0
    for cloth in rigid_scene.cloths:
        slices.append(slice(v_start, v_start + len(cloth.verts)))
        v_start += len(cloth.verts)
    # The frame of CGQ's root is the principal frame of the box, a constant rotation of the frame of its link
    axes = scenes.quat_to_R(box.quat).T @ scenes.quat_to_R(cgq_rigid[0][3:7])
    box_radius = 0.5 * float(np.linalg.norm(box.size))

    def get_positions():
        return np.concatenate([entity.get_verts_pos().cpu().numpy() for entity in entities])

    def box_error(state) -> float:
        """The largest surface displacement |dt| + r |dtheta| of the box from the pose of CGQ's root."""
        if box.is_fixed:
            return 0.0
        rotation = gu.quat_to_R(box_entity.get_quat().cpu().numpy()) @ axes
        relative = rotation.T @ scenes.quat_to_R(state[3:7])
        angle = np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
        return float(np.linalg.norm(box_entity.get_pos().cpu().numpy() - state[0:3]) + box_radius * angle)

    def set_box(state) -> None:
        if box.is_fixed:
            return
        rotation = scenes.quat_to_R(state[3:7]) @ axes.T
        box_entity.set_qpos(np.concatenate([state[0:3], gu.R_to_quat(rotation)]))
        box_entity.set_dofs_velocity(np.concatenate([state[7:10], rotation.T @ state[10:13]]))

    def set_proxy(state) -> None:
        """Offset the proxy of the box from the inertial frame of its link as CGQ's proxy is offset from its root, the
        principal axes of both sides possibly differing by a permutation."""
        if box.is_fixed:
            return
        proxies_state = newton.data_manager.proxies_state
        quat = qd_to_numpy(proxies_state.quat)
        relative = scenes.quat_to_R(state[16:20]) @ scenes.quat_to_R(state[3:7]).T
        quat[0, 0] = gu.R_to_quat(relative @ scenes.quat_to_R(quat[0, 0]))
        pos = qd_to_numpy(proxies_state.pos)
        pos[0, 0] = state[13:16]
        proxies_state.quat.from_numpy(quat)
        proxies_state.pos.from_numpy(pos)

    init_error = float(np.abs(get_positions() - cgq_positions[0]).max())
    replay_errors, box_errors, step_ms, pairs_rows, counts = [], [], [], [], []
    for k in range(rigid_scene.n_frames):
        for entity, part in zip(entities, slices):
            entity.set_verts_pos(cgq_positions[k][part])
            entity.set_verts_vel(cgq_velocities[k][part])
        set_box(cgq_rigid[k])
        init_proxies(scene)
        set_proxy(cgq_rigid[k])
        genesis_pairs = genesis_active_pairs(scene)
        row = {}
        for ch in ("pt", "ee"):
            offsets = reference[f"{ch}_offsets"]
            row[ch] = compare_pairs(
                genesis_pairs[ch],
                reference[f"{ch}_stencil"][offsets[k] : offsets[k + 1]],
                reference[f"{ch}_flag"][offsets[k] : offsets[k + 1]],
                ch == "ee",
            )
        pairs_rows.append(row)
        counters.take(scene)
        t = time.perf_counter()
        scene.step()
        step_ms.append(1e3 * (time.perf_counter() - t))
        counts.append(counters.take(scene))
        replay_errors.append(float(np.abs(get_positions() - cgq_positions[k + 1]).max()))
        box_errors.append(box_error(cgq_rigid[k + 1]))

    scene.reset()
    set_initial_spin()
    trajectory_errors, box_trajectory_errors, n_intersections = [], [], []
    for k in range(rigid_scene.n_frames):
        scene.step()
        trajectory_errors.append(float(np.abs(get_positions() - cgq_positions[k + 1]).max()))
        box_trajectory_errors.append(box_error(cgq_rigid[k + 1]))
        publish_start_point(scene)
        n_intersections.append(len(newton.ipc_contact.find_intersections(newton.ipc_state, newton.ipc_info)))

    replay_errors = np.array(replay_errors)
    totals = {
        ch: {key: int(sum(row[ch][key] for row in pairs_rows)) for key in pairs_rows[0][ch]} for ch in ("pt", "ee")
    }
    result = {
        "init_error": init_error,
        "replay_max_error": float(replay_errors.max()),
        "replay_worst_frame": int(replay_errors.argmax()),
        "replay_frames_above_bound": int((replay_errors > bound).sum()),
        "replay_errors": replay_errors.tolist(),
        "trajectory_errors": trajectory_errors,
        "trajectory_max_error": float(max(trajectory_errors)),
        "box_replay_errors": box_errors,
        "box_replay_max_error": float(max(box_errors)),
        "box_replay_frames_above_bound": int((np.array(box_errors) > bound).sum()),
        "box_trajectory_errors": box_trajectory_errors,
        "box_trajectory_max_error": float(max(box_trajectory_errors)),
        "genesis_rest_box_y": float(box_entity.get_pos().cpu().numpy()[1]),
        "cgq_rest_box_y": float(cgq_rigid[-1][1]),
        "newton_counts": compare_counts(counts, reference) if "ls_iters" in reference else None,
        "genesis_step_ms": step_ms,
        "genesis_step_ms_median": float(np.median(step_ms[1:])),
        "cgq_step_ms_median": float(np.median(reference["step_ms"][1:])),
        "pairs": pairs_rows,
        "pairs_totals": totals,
        "pairs_frames_differing": int(
            sum(any(row[ch]["only_genesis"] or row[ch]["only_cgq"] for ch in ("pt", "ee")) for row in pairs_rows)
        ),
        "trajectory_intersections": int(sum(n_intersections)),
        "genesis_rest_min_y": float(get_positions()[:, 1].min()),
        "cgq_rest_min_y": float(cgq_positions[-1][:, 1].min()),
    }
    summary["scenes"][rigid_scene.name] = result
    print(
        f"{rigid_scene.name} [{args.backend}]: replay max {result['replay_max_error']:.3e} (bound {bound:.1e}, "
        f"{result['replay_frames_above_bound']} frames above, worst frame {result['replay_worst_frame']}), "
        f"trajectory max {result['trajectory_max_error']:.3e}, box replay max {result['box_replay_max_error']:.3e} "
        f"({result['box_replay_frames_above_bound']} frames above), box trajectory max "
        f"{result['box_trajectory_max_error']:.3e}, step {result['genesis_step_ms_median']:.2f} ms, "
        f"pairs {totals} ({result['pairs_frames_differing']} frames differ), intersections "
        f"{result['trajectory_intersections']}"
        + (
            f", newton/ls/pcg totals {result['newton_counts']['totals']}, "
            f"{len(result['newton_counts']['flagged_frames'])} frames above 1.5x CGQ's Newton iterations"
            if result["newton_counts"] is not None
            else ""
        ),
        flush=True,
    )
    scene.destroy()

if not args.dump_mesh:
    out = results_dir / f"genesis_{args.backend}{args.tag}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"PASS: {all(s['replay_frames_above_bound'] == 0 for s in summary['scenes'].values())} -> {out}")
