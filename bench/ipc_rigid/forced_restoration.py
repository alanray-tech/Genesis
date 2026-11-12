"""Force the restoration of the rigid contact proxies in Genesis by starving the line search, on the free box dropped onto
a cloth lying on the ground (box_on_cloth of rigid_scenes.py, in double precision).

For every cap of line-search trials, the scene runs its whole horizon and reports the restoration entries, the dual
updates and hard probes that follow them, the Newton iterations, the errno of the solve, the edges crossing a triangle
on any frame, the largest surface residual of the proxy at the committed state against the solve tolerance, and where
the box comes to rest against the default cap.

    python bench/ipc_rigid/forced_restoration.py --backend gpu|cpu [--caps 1 2 3 12]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ipc_cloth"))

import rigid_scenes as scenes  # noqa: E402
from contact_probe import publish_start_point  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--caps", type=int, nargs="*", default=[1, 2, 3, 12])
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")

import genesis.engine.solvers.newton_solver as newton_solver  # noqa: E402

counts = {"entries": 0, "dual_updates": 0, "hard_probes": 0, "newton": 0}
end_iteration = newton_solver.kernel_end_iteration


def counted_end_iteration(newton_envs, newton_loops, proxies_envs, *args):
    is_active = bool(qd_to_numpy(newton_envs.is_active)[0])
    is_entered = bool(qd_to_numpy(proxies_envs.is_restoration_entered)[0])
    is_hard_probe = bool(qd_to_numpy(proxies_envs.is_hard_probe)[0])
    end_iteration(newton_envs, newton_loops, proxies_envs, *args)
    if is_active:
        counts["newton"] += 1
        counts["entries"] += is_entered
        counts["dual_updates"] += bool(qd_to_numpy(proxies_envs.is_dual_update)[0])
        counts["hard_probes"] += bool(qd_to_numpy(proxies_envs.is_hard_probe)[0]) and not is_hard_probe


newton_solver.kernel_end_iteration = counted_end_iteration

rigid_scene = next(s for s in scenes.make_scenes(16) if s.name == "box_on_cloth")
(cloth_spec,) = rigid_scene.cloths
(box,) = rigid_scene.boxes
mesh_path = Path(args.data) / "forced_restoration_cloth.obj"
mesh_path.write_text(
    trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth_spec.verts, cloth_spec.tris, process=False), digits=17)
)
solve_tolerance = min(scenes.VELOCITY_TOL * scenes.DT, 0.01 * scenes.D_HAT)
summary = {"backend": args.backend, "solve_tolerance": solve_tolerance, "caps": {}}
for cap in args.caps:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
        newton_options=gs.options.NewtonOptions(velocity_tol=scenes.VELOCITY_TOL, line_search_max_iterations=cap),
        ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
        show_viewer=False,
    )
    cloth = scene.add_entity(
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
    newton = scene.sim.newton_solver
    proxies_state = newton.data_manager.proxies_state
    for key in counts:
        counts[key] = 0
    residuals, n_intersections, errnos, error = [], 0, 0, None
    for _ in range(rigid_scene.n_frames):
        try:
            scene.step()
        except gs.GenesisException as exception:
            error = str(exception)
            break
        errnos |= int(qd_to_numpy(newton._errno)[0])
        link_pos, link_quat = box_entity.get_pos().cpu().numpy(), box_entity.get_quat().cpu().numpy()
        inertial_pos = qd_to_numpy(proxies_state.inertial_pos)[0, 0]
        inertial_R = gu.quat_to_R(qd_to_numpy(proxies_state.inertial_quat)[0, 0])
        link_R = gu.quat_to_R(link_quat)
        relative = (link_R @ inertial_R).T @ gu.quat_to_R(qd_to_numpy(proxies_state.quat)[0, 0])
        angle = np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
        offset = qd_to_numpy(proxies_state.pos)[0, 0] - (link_pos + link_R @ inertial_pos)
        residuals.append(float(np.linalg.norm(offset) + qd_to_numpy(proxies_state.surface_radius)[0, 0] * angle))
        publish_start_point(scene)
        n_intersections += len(newton.ipc_contact.find_intersections(newton.ipc_state, newton.ipc_info))
    result = {
        **counts,
        "errno": errnos,
        "error": error,
        "n_frames_run": len(residuals),
        "intersections": n_intersections,
        "commit_residual_max": max(residuals),
        "commit_frames_above_tolerance": int((np.array(residuals) > solve_tolerance).sum()),
        "rest_box_pos": box_entity.get_pos().cpu().numpy().tolist(),
        "rest_box_quat": box_entity.get_quat().cpu().numpy().tolist(),
        "cloth_min_y": float(cloth.get_verts_pos().cpu().numpy()[..., 1].min()),
    }
    summary["caps"][cap] = result
    print(f"cap {cap}: {result}", flush=True)
    scene.destroy()

reference = summary["caps"][max(args.caps)]
for cap, result in summary["caps"].items():
    result["rest_box_offset_vs_largest_cap"] = float(
        np.linalg.norm(np.array(result["rest_box_pos"]) - np.array(reference["rest_box_pos"]))
    )
out = Path(args.results) / f"forced_restoration_{args.backend}.json"
out.write_text(json.dumps(summary, indent=1))
print(f"-> {out}")
