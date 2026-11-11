"""Replay every frame of the CGQ cloth runs in Genesis, and run the whole trajectory, in double precision.

Replay: the state of CGQ at frame k (positions and velocities) is loaded into Genesis, which steps once, and its
positions are compared with CGQ's frame k + 1. The bound is 2 * velocity_tol * dt, both solves stopping within
velocity_tol * dt of the solution. Trajectory: Genesis runs from frame 0 on its own, compared with CGQ frame by frame.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs

sys.path.insert(0, str(Path(__file__).parent))

import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
data_dir, results_dir = Path(args.data), Path(args.results)
results_dir.mkdir(parents=True, exist_ok=True)

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")
bound = 2.0 * scenes.VELOCITY_TOL * scenes.DT
summary = {"backend": args.backend, "n": args.n, "bound": bound, "scenes": {}}

for cloth_scene in scenes.make_scenes(args.n):
    reference = np.load(data_dir / f"cgq_{cloth_scene.name}_n{args.n}.npz")
    cgq_positions, cgq_velocities = reference["positions"], reference["velocities"]
    mesh_path = data_dir / f"{cloth_scene.name}_n{args.n}.obj"
    mesh_path.write_text(
        trimesh.exchange.obj.export_obj(
            trimesh.Trimesh(cloth_scene.verts, cloth_scene.tris, process=False), digits=17
        )
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
        newton_options=gs.options.NewtonOptions(
            velocity_tol=scenes.VELOCITY_TOL, max_iterations=1024, line_search_max_iterations=12
        ),
        ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
        show_viewer=False,
    )
    if cloth_scene.has_ground:
        scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
    cloth = scene.add_entity(
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
    scene.build()
    init_error = float(np.abs(cloth.get_verts_pos().cpu().numpy() - cgq_positions[0]).max())
    if cloth_scene.pinned.any():
        cloth.set_vertex_constraints(verts_idx_local=np.flatnonzero(cloth_scene.pinned))

    # One-step replay of every frame of CGQ, in order, so that the friction pairs of a frame are those of the step
    # before it
    replay_errors, step_ms = [], []
    for k in range(cloth_scene.n_frames):
        cloth.set_verts_pos(cgq_positions[k])
        cloth.set_verts_vel(cgq_velocities[k])
        t = time.perf_counter()
        scene.step()
        positions = cloth.get_verts_pos().cpu().numpy()
        step_ms.append(1e3 * (time.perf_counter() - t))
        replay_errors.append(float(np.abs(positions - cgq_positions[k + 1]).max()))

    # The whole trajectory, from the initial state
    scene.reset()
    if cloth_scene.pinned.any():
        cloth.set_vertex_constraints(verts_idx_local=np.flatnonzero(cloth_scene.pinned))
    trajectory_errors = []
    for k in range(cloth_scene.n_frames):
        scene.step()
        positions = cloth.get_verts_pos().cpu().numpy()
        trajectory_errors.append(float(np.abs(positions - cgq_positions[k + 1]).max()))

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
        result["genesis_trajectory_min_y"] = float(positions[:, 1].min())
    summary["scenes"][cloth_scene.name] = result
    print(
        f"{cloth_scene.name} [{args.backend}]: replay max {result['replay_max_error']:.3e} (bound {bound:.1e}, "
        f"{result['replay_frames_above_bound']} frames above, worst frame {result['replay_worst_frame']}), "
        f"trajectory max {result['trajectory_max_error']:.3e}, step {result['genesis_step_ms_median']:.2f} ms"
    )
    scene.destroy()

out = results_dir / f"genesis_{args.backend}_n{args.n}.json"
out.write_text(json.dumps(summary, indent=1))
print(f"PASS: {all(s['replay_frames_above_bound'] == 0 for s in summary['scenes'].values())} -> {out}")
