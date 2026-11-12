"""Genesis side of the proof of CGQ's Genesis rigid model, in double precision (see genesis_model_scenes.py).

Every scene runs in Genesis from its first frame. Per frame k the rigid entity's qpos and dof velocities and its controls
(mode, position, velocity, force per dof) are saved before the step, and after it the quantities of the substep start
that the Newton solve took: the mass matrix M' (armature and implicit damping included), qf_smooth, qf_bias and the
constraint rows (D, aref, Jacobian over the entity's dofs). The entity's model is saved once: per dof Genesis's
dofs_info (armature, damping, stiffness, invweight, act_gain, act_bias, force_range, limit, frictionloss) and its
joint's sol_params, its joint equalities, its joint names and gs.EPS, and for a free root the link frame at the first
frame. genesis_model_cgq.py loads all of it into CGQ.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

import genesis as gs
from genesis.utils.misc import get_assets_dir, qd_to_numpy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ipc_cloth"))

import franka_scene as fs  # noqa: E402
import genesis_model_scenes as gms  # noqa: E402
import rigid_scenes as scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--fast-math", type=int, default=1)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
args = parser.parse_args()
gms.DATA.mkdir(parents=True, exist_ok=True)

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning", debug=not args.fast_math)
from newton_counters import NewtonCounters  # noqa: E402


def add_cloth(scene, verts, tris, name, bending, E, rho, thickness, shear, bending_E):
    path = gms.DATA / f"{name}.obj"
    path.write_text(trimesh.exchange.obj.export_obj(trimesh.Trimesh(verts, tris, process=False), digits=17))
    return scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.Cloth(
            E=E, rho=rho, thickness=thickness, shear_modulus=shear, bending=bending, bending_E=bending_E
        ),
    )


def build(name):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
        newton_options=gs.options.NewtonOptions(velocity_tol=args.velocity_tol, max_iterations=1024),
        ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
        show_viewer=False,
    )
    if name == "box_spin":
        (rigid_scene,) = [s for s in scenes.make_scenes(16) if s.name == "box_spin"]
        (cloth,) = rigid_scene.cloths
        add_cloth(
            scene,
            cloth.verts,
            cloth.tris,
            "box_spin_cloth",
            rigid_scene.bending,
            scenes.YOUNGS_MODULUS,
            scenes.MASS_DENSITY,
            scenes.THICKNESS,
            scenes.SHEAR_MODULUS,
            scenes.BENDING_YOUNGS_MODULUS,
        )
        scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
        (box,) = rigid_scene.boxes
        entity = scene.add_entity(
            morph=gs.morphs.Box(size=box.size, pos=box.pos, quat=box.quat, fixed=box.is_fixed),
            material=gs.materials.Rigid(rho=box.density, coup_friction=gs.materials.FEM.Cloth().friction_mu),
        )
        scene.build()
        entity.set_dofs_velocity(np.concatenate([np.zeros(3), scenes.quat_to_R(box.quat).T @ np.array(box.omega)]))
    else:
        verts, tris = gms.far_cloth(shift_y=fs.GROUND_H + 0.02)
        add_cloth(
            scene,
            verts,
            tris,
            "franka_swing_cloth",
            "hinge",
            fs.YOUNGS_MODULUS,
            fs.MASS_DENSITY,
            fs.CLOTH_THICKNESS,
            fs.SHEAR_MODULUS,
            fs.BENDING_YOUNGS_MODULUS,
        )
        scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, fs.GROUND_H, 0.0), normal=(0.0, 1.0, 0.0)))
        franka_path = Path(get_assets_dir()) / "xml" / "franka_emika_panda" / "panda.xml"
        entity = scene.add_entity(
            morph=gs.morphs.MJCF(file=str(franka_path), pos=fs.BASE_POS, quat=fs.BASE_QUAT),
            material=gs.materials.Rigid(coup_friction=fs.FRICTION),
        )
        scene.build()
        entity.set_qpos(fs.HOME_QPOS)
        entity.set_dofs_force_range(-fs.ARM_FORCE_RANGE, fs.ARM_FORCE_RANGE, dofs_idx_local=range(7))
        entity.control_dofs_position(gms.SWING_TARGET)
    assert scene.sim.rigid_solver.is_solved_in_global_newton
    return scene, entity


for name in gms.N_FRAMES:
    if args.only and name not in args.only:
        continue
    scene, entity = build(name)
    rigid = scene.sim.rigid_solver
    counters = NewtonCounters(gs.options.NewtonOptions().pcg_max_iterations)
    dofs = np.arange(entity.dof_start, entity.dof_start + entity.n_dofs)
    state = rigid.dyn_state.dofs
    cs = rigid.constraint_solver.constraint_state
    record = {key: [] for key in ("qpos", "vel", "ctrl_mode", "ctrl_pos", "ctrl_vel", "ctrl_force")}
    record.update({key: [] for key in ("mass", "qf_smooth", "qf_bias", "rows_D", "rows_aref", "rows_jac", "counts")})
    for k in range(gms.N_FRAMES[name]):
        record["qpos"].append(entity.get_qpos().cpu().numpy())
        record["vel"].append(entity.get_dofs_velocity().cpu().numpy())
        for key in ("ctrl_mode", "ctrl_pos", "ctrl_vel", "ctrl_force"):
            record[key].append(qd_to_numpy(getattr(state, key))[dofs, 0])
        counters.take(scene)
        scene.step()
        record["counts"].append(counters.take(scene))
        record["mass"].append(qd_to_numpy(rigid.rigid_info.mass_mat)[np.ix_(dofs, dofs)][..., 0])
        record["qf_smooth"].append(qd_to_numpy(state.qf_smooth)[dofs, 0])
        record["qf_bias"].append(qd_to_numpy(state.qf_bias)[dofs, 0])
        n_con = int(qd_to_numpy(cs.n_constraints)[0])
        D = np.zeros(len(dofs))
        aref = np.zeros(len(dofs))
        jac = np.zeros((len(dofs), len(dofs)))
        D[:n_con] = qd_to_numpy(cs.efc_D)[:n_con, 0]
        aref[:n_con] = qd_to_numpy(cs.aref)[:n_con, 0]
        jac[:n_con] = qd_to_numpy(cs.jac)[:n_con][:, dofs, 0]
        record["rows_D"].append(D)
        record["rows_aref"].append(aref)
        record["rows_jac"].append(jac)
    record["qpos"].append(entity.get_qpos().cpu().numpy())
    record["vel"].append(entity.get_dofs_velocity().cpu().numpy())
    model = gms.entity_model(scene, entity)
    path = gms.DATA / f"genesis_{name}.npz"
    np.savez(
        path,
        **{key: np.array(value) for key, value in record.items()},
        **{f"model_{key}": value for key, value in model.items()},
    )
    counts = np.array(record["counts"])
    print(
        f"{name}: {len(counts)} frames, Genesis newton/ls/pcg totals {counts.sum(axis=0).tolist()}, rows per frame max "
        f"{int(max((np.abs(j).sum(axis=1) > 0).sum() for j in record['rows_jac']))}, joints {model['joint_names'].tolist()}"
        f" -> {path}",
        flush=True,
    )
    scene.destroy()
