"""CGQ side of the proof of CGQ's Genesis rigid model (rigid_forest/model = "genesis") against Genesis (run with CGQ's
own venv, after genesis_model_genesis.py; see genesis_model_scenes.py).

Every scene is built in CGQ's Genesis mode with Genesis's per-dof numbers, joint equalities and EPS loaded. Every frame
k of Genesis's run is replayed for one step: Genesis's qpos, dof velocities and controls of frame k are set, CGQ steps,
and are compared
- proof 0: the substep-start quantities, CGQ's M' / f / qf_bias against Genesis's mass_mat / qf_smooth / qf_bias of frame
  k (relative to the largest entry),
- proof (a): CGQ's state after the step against Genesis's frame k + 1, as the largest surface displacement of a body,
  |dt| + r |dtheta| (r the radius of the body's contact proxy), against 2 velocity_tol dt, and the dof velocities,
with the Newton iterations of both sides per frame.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import franka_scene as fs  # noqa: E402
import genesis_model_scenes as gms  # noqa: E402
import rigid_scenes as scenes  # noqa: E402

quat_to_R = scenes.quat_to_R
quat_mul = gms.quat_mul


def build(name, velocity_tol):
    """The CGQ scene of a proof scene in the Genesis rigid model (CGQ on sys.path)."""
    from qipc import Cloth, RigidBody, Scene, halfplane, trimesh
    from qipc.geometry import ground

    scene = Scene(
        dt=scenes.DT,
        gravity=scenes.GRAVITY,
        **{
            "contact/enable": 1,
            "contact/d_hat": scenes.D_HAT,
            "newton/velocity_tol": velocity_tol,
            "newton/max_iter": 1024,
            "line_search/max_iter": 12,
            "rigid_forest/model": "genesis",
        },
    )
    if name == "box_spin":
        (rigid_scene,) = [s for s in scenes.make_scenes(16) if s.name == "box_spin"]
        (cloth,) = rigid_scene.cloths
        geo = trimesh(cloth.verts, cloth.tris)
        Cloth().apply_to(
            geo,
            youngs_modulus=scenes.YOUNGS_MODULUS,
            shear_modulus=scenes.SHEAR_MODULUS,
            mass_density=scenes.MASS_DENSITY,
            thickness=scenes.THICKNESS,
            bending=rigid_scene.bending,
            bending_youngs_modulus=scenes.BENDING_YOUNGS_MODULUS,
        )
        scene.geometries.create("cloth", geo)
        scene.geometries.create("ground", halfplane(P=scenes.GROUND_POS, N=scenes.GROUND_NORMAL))
        mesh = np.load(gms.SOURCE_DATA / "rigid_mesh_box_spin.npz")
        (box,) = rigid_scene.boxes
        mass, com, inertia = box.mass_properties()
        rigid_geo = trimesh(mesh["verts"], mesh["tris"].astype(np.int32))
        RigidBody().apply_to(rigid_geo, mass=mass, center_of_mass=com, inertia=inertia, minimal_coordinates=True)
        scene.geometries.create("rigid", rigid_geo)
    else:
        verts, tris = gms.far_cloth(shift_y=fs.GROUND_H + 0.02)
        geo = trimesh(verts, tris)
        Cloth().apply_to(
            geo,
            youngs_modulus=fs.YOUNGS_MODULUS,
            shear_modulus=fs.SHEAR_MODULUS,
            mass_density=fs.MASS_DENSITY,
            thickness=fs.CLOTH_THICKNESS,
            bending="hinge",
            bending_youngs_modulus=fs.BENDING_YOUNGS_MODULUS,
        )
        scene.geometries.create("cloth", geo)
        scene.geometries.create("ground", ground(height=fs.GROUND_H))
        scene.add_mjcf(
            gms.copy_franka_assets(),
            link_type="rigid",
            minimal_coordinates=True,
            fix_base=True,
            position=fs.BASE_POS,
            rotation=fs.BASE_EULER_DEG,
        )
    scene.init()
    return scene


def dof_map(model, genesis_names, name) -> np.ndarray:
    """CGQ's dof of every Genesis dof."""
    if name == "box_spin":
        return np.arange(6)
    mapping = []
    for joint in genesis_names:
        matches = [i for i, dof in enumerate(model.dof_names) if dof.split("/")[-1].split(":")[-1] == joint]
        assert len(matches) == 1, f"Genesis joint {joint} matches CGQ dofs {matches} of {model.dof_names}"
        mapping.append(matches[0])
    return np.array(mapping)


def load_model(scene, g, perm):
    model = scene.rigid_genesis_model
    model.set_dofs_params(
        perm,
        armature=g["model_armature"],
        damping=g["model_damping"],
        stiffness=g["model_stiffness"],
        act_gain=g["model_act_gain"],
        act_bias=g["model_act_bias"],
        force_range=g["model_force_range"],
        limit=g["model_limit"],
        invweight=g["model_invweight"],
        sol_params=g["model_sol_params"],
    )
    model.set_joint_equalities(
        [(int(perm[int(e[0])]), int(perm[int(e[1])]), e[2:7], e[7:14]) for e in g["model_equalities"]]
    )
    model.set_eps(float(g["model_eps"]))
    return model


def set_controls(model, g, k, perm):
    for mode, setter, key in (
        (0, model.control_dofs_position, "ctrl_pos"),
        (1, model.control_dofs_velocity, "ctrl_vel"),
        (2, model.control_dofs_force, "ctrl_force"),
    ):
        selected = np.nonzero(g["ctrl_mode"][k] == mode)[0]
        if len(selected):
            setter(g[key][k][selected], dofs=perm[selected])
    position_dofs = np.nonzero(g["ctrl_mode"][k] == 0)[0]
    # A position-controlled dof also reads its velocity target, CGQ's left at zero
    assert np.all(g["ctrl_vel"][k][position_dofs] == 0.0)


def cgq_qpos(g_qpos, perm, n_qs, name):
    """Genesis's qpos in CGQ's order (a free root's 7 entries first, as the box's only tree)."""
    if name == "box_spin":
        return np.asarray(g_qpos)
    out = np.zeros(n_qs)
    out[perm] = g_qpos
    return out


def cgq_vel(g_vel, perm):
    out = np.zeros(len(perm))
    out[perm] = g_vel
    return out


def replay(name, velocity_tol) -> dict:
    """Replay every frame of Genesis's run of a proof scene for one step in CGQ (see the module docstring)."""
    bound = 2.0 * velocity_tol * scenes.DT
    g = np.load(gms.DATA / f"genesis_{name}.npz")
    scene = build(name, velocity_tol)
    model = scene.rigid_genesis_model
    rigid = scene.rigid_body
    native_rigid = rigid._native
    forest = scene._rigid_joint_forest
    perm = dof_map(model, [str(j) for j in g["model_joint_names"]], name)
    load_model(scene, g, perm)
    pairs = np.asarray(rigid.host("contact_proxy_pairs")).reshape(-1, 2)
    x_bar = np.asarray(rigid.host("x_bar")).reshape(-1, 3)
    body_id = np.asarray(rigid.host("body_id")).reshape(-1)
    radius = {int(m): float(np.linalg.norm(x_bar[body_id == p], axis=1).max()) for m, p in pairs}
    if name == "box_spin":
        (root,) = [int(r) for r in forest.tree_roots]
        q_body = native_rigid.quat_view_t()[root].cpu().numpy()
        t_body = native_rigid.t_view_t()[root].cpu().numpy()
        p_link, q_link = g["qpos"][0][:3], g["qpos"][0][3:]
        rotation = quat_mul(q_body * np.array([1.0, -1.0, -1.0, -1.0]), q_link)
        origin = quat_to_R(q_body).T @ (p_link - t_body)
        model.set_root_link_frame(root, rotation, origin)
    sizes = [model.tree_dofs(t).stop - model.tree_dofs(t).start for t in range(len(forest.tree_roots))]
    (tree,) = [t for t, n in enumerate(sizes) if n == len(perm)]

    def poses():
        return native_rigid.t_view_t().cpu().numpy().copy(), native_rigid.quat_view_t().cpu().numpy().copy()

    n_frames = gms.N_FRAMES[name]
    rows = []
    stepped = None
    for k in range(n_frames + 1):
        model.set_state(cgq_qpos(g["qpos"][k], perm, model.n_qs, name), cgq_vel(g["vel"][k], perm))
        if stepped is not None:
            t_ref, quat_ref = poses()
            t_cgq, quat_cgq, vel_cgq, counts = stepped
            error = 0.0
            for body, r in radius.items():
                relative = quat_to_R(quat_cgq[body]).T @ quat_to_R(quat_ref[body])
                angle = np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
                error = max(error, float(np.linalg.norm(t_cgq[body] - t_ref[body]) + r * angle))
            rows[-1].update(
                {
                    "state_error": error,
                    "velocity_error": float(np.abs(vel_cgq - cgq_vel(g["vel"][k], perm)).max()),
                    "cgq_newton": counts[0],
                    "cgq_ls": counts[1],
                    "cgq_pcg": counts[2],
                }
            )
        if k == n_frames:
            break
        set_controls(model, g, k, perm)
        scene.step()
        M = model.mass_matrix(tree)[np.ix_(perm, perm)]
        f = model.smooth_force[model.tree_dofs(tree)][perm]
        bias = model.bias_force[model.tree_dofs(tree)][perm]
        scale_M = np.abs(g["mass"][k]).max()
        scale_f = max(np.abs(g["qf_smooth"][k]).max(), 1e-300)
        rows.append(
            {
                "frame": k,
                "mass_rel_error": float(np.abs(M - g["mass"][k]).max() / scale_M),
                "qf_smooth_rel_error": float(np.abs(f - g["qf_smooth"][k]).max() / scale_f),
                "qf_bias_rel_error": float(
                    np.abs(bias - g["qf_bias"][k]).max() / max(np.abs(g["qf_bias"][k]).max(), 1e-300)
                ),
                "genesis_newton": int(g["counts"][k][0]),
                "genesis_ls": int(g["counts"][k][1]),
                "genesis_pcg": int(g["counts"][k][2]),
                "omega_dt": float(np.linalg.norm(g["vel"][k][3:6]) * scenes.DT) if name == "box_spin" else None,
            }
        )
        t_cgq, quat_cgq = poses()
        stepped = (
            t_cgq,
            quat_cgq,
            model.dofs_vel.cpu().numpy().copy(),
            (scene.solver.newton_iters, scene.solver.total_ls_iters, scene.solver.total_pcg_iters),
        )
    errors = np.array([row["state_error"] for row in rows])
    result = {
        "frames": rows,
        "state_max_error": float(errors.max()),
        "state_worst_frame": int(errors.argmax()),
        "state_frames_above_bound": int((errors > bound).sum()),
        "velocity_max_error": float(max(row["velocity_error"] for row in rows)),
        "mass_max_rel_error": float(max(row["mass_rel_error"] for row in rows)),
        "qf_smooth_max_rel_error": float(max(row["qf_smooth_rel_error"] for row in rows)),
        "qf_bias_max_rel_error": float(max(row["qf_bias_rel_error"] for row in rows)),
        "newton_totals": {
            side: int(sum(row[f"{side}_newton"] for row in rows)) for side in ("genesis", "cgq")
        },
    }
    print(
        f"{name}: state max {result['state_max_error']:.3e} (bound {bound:.1e}, {result['state_frames_above_bound']} "
        f"frames above, worst frame {result['state_worst_frame']}), velocity max {result['velocity_max_error']:.3e}, "
        f"M' rel {result['mass_max_rel_error']:.2e}, qf_smooth rel {result['qf_smooth_max_rel_error']:.2e}, "
        f"qf_bias rel {result['qf_bias_max_rel_error']:.2e}, Newton totals {result['newton_totals']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cgq", required=True, help="the CGQ checkout, for its packages")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    sys.path.insert(0, args.cgq)
    summary = {"velocity_tol": args.velocity_tol, "bound": 2.0 * args.velocity_tol * scenes.DT, "scenes": {}}
    for name in gms.N_FRAMES:
        if not args.only or name in args.only:
            summary["scenes"][name] = replay(name, args.velocity_tol)
    gms.RESULTS.mkdir(parents=True, exist_ok=True)
    out = gms.RESULTS / f"genesis_model_cgq{args.tag}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"PASS: {all(s['state_frames_above_bound'] == 0 for s in summary['scenes'].values())} -> {out}")


if __name__ == "__main__":
    main()
