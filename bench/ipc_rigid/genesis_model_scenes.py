"""Scenes of the proof of CGQ's Genesis rigid model (rigid_forest/model = "genesis") against Genesis, shared by
genesis_model_genesis.py (Genesis side, dumps) and genesis_model_cgq.py (CGQ side, one-step replays).

box_spin: the free box of rigid_scenes.py spinning in the air at 12.5 rad/s, the cloth lying on the ground far from it
    (Genesis solves the box in the Newton solve only with a cloth in the scene), no contact.
franka_swing: the fixed-base Franka of franka_scene.py swinging from its home pose under Genesis's own joint actuators
    toward a target 0.4-0.6 rad away on five joints, its fingers held open (see FINGER_OPEN_CTRL), a cloth lying on the
    ground far from it, no contact.

The Franka of both sides is the MJCF franka_genesis.py --dump writes (Genesis's bodies, joints and inertials, its proxy
meshes as collision meshes), copied once into data_gm/ (copy_franka_assets) so that a later dump of the parity case 3
cannot change it under a run.
"""

import shutil
from pathlib import Path

import numpy as np

import franka_scene as fs
import rigid_scenes as scenes

DATA = Path(__file__).parent / "data_gm"
RESULTS = Path(__file__).parent / "results"
SOURCE_DATA = Path(__file__).parent / "data"

N_FRAMES = {"box_spin": 30, "franka_swing": 100}
FAR_CLOTH_OFFSET = np.array([3.0, 0.0, 0.0])
# The fingers' actuator of Genesis (its approximation of the MJCF tendon) holds them open at the control 255, where
# 0.0157 * 255 / 100 = 0.04: a control of 0.04 would close them until their proxies meet, a rigid-rigid contact that
# CGQ's IPC takes and Genesis leaves to its native contact
FINGER_OPEN_CTRL = 255.0
SWING_TARGET = np.concatenate(
    [np.array(fs.HOME_QPOS[:7]) + np.array([0.5, 0.4, -0.6, 0.5, 0.0, -0.4, 0.0]), [FINGER_OPEN_CTRL] * 2]
)


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a (x) b of (w, x, y, z) quaternions."""
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def copy_franka_assets() -> Path:
    """Copy the Franka MJCF of franka_genesis.py --dump and its proxy meshes into data_gm/, the mesh paths rewritten."""
    DATA.mkdir(parents=True, exist_ok=True)
    text = (SOURCE_DATA / "franka_proxies.xml").read_text()
    for mesh in SOURCE_DATA.glob("franka_proxy_*.obj"):
        shutil.copyfile(mesh, DATA / mesh.name)
        text = text.replace(str(mesh), str(DATA / mesh.name))
    path = DATA / "franka_proxies.xml"
    path.write_text(text)
    return path


def entity_model(scene, entity) -> dict:
    """Genesis side: the rigid model of an entity as Genesis holds it, which genesis_model_cgq.load_model loads into
    CGQ: per dof its dofs_info and its joint's sol_params, its joint names, its joint equalities (dof1, dof2, polynomial
    (5), sol_params (7)), gs.EPS and its qpos0."""
    import genesis as gs
    from genesis.utils.misc import qd_to_numpy

    rigid = scene.sim.rigid_solver
    dofs = np.arange(entity.dof_start, entity.dof_start + entity.n_dofs)
    info = rigid.dyn_info

    def dof_info(name):
        values = qd_to_numpy(getattr(info.dofs, name))
        return values.reshape(values.shape[0], -1)[dofs].squeeze()

    joints_type = qd_to_numpy(info.joints.type).reshape(-1)
    joints_dof_start = qd_to_numpy(info.joints.dof_start).reshape(-1)
    joints_sol = qd_to_numpy(info.joints.sol_params).reshape(len(joints_type), -1)
    sol_params = np.zeros((len(dofs), 7))
    joint_names = []
    for joint in entity.joints:
        for k in range(joint.n_dofs):
            sol_params[joint.dof_start + k - entity.dof_start] = joints_sol[joint.idx]
            joint_names.append(joint.name)
    model = {
        "armature": dof_info("armature"),
        "damping": dof_info("damping"),
        "stiffness": dof_info("stiffness"),
        "invweight": dof_info("invweight"),
        "frictionloss": dof_info("frictionloss"),
        "act_gain": dof_info("act_gain"),
        "act_bias": dof_info("act_bias").reshape(len(dofs), 3),
        "force_range": dof_info("force_range").reshape(len(dofs), 2),
        "limit": dof_info("limit").reshape(len(dofs), 2),
        "sol_params": sol_params,
        "joint_names": np.array(joint_names),
        "eps": np.array(gs.EPS),
        "qpos0": qd_to_numpy(rigid.rigid_info.qpos0).reshape(-1)[entity.q_start : entity.q_end],
    }
    equalities = []
    eq_type = qd_to_numpy(info.equalities.eq_type)[:, 0]
    eq_obj1 = qd_to_numpy(info.equalities.eq_obj1id)[:, 0]
    eq_obj2 = qd_to_numpy(info.equalities.eq_obj2id)[:, 0]
    eq_data = qd_to_numpy(info.equalities.eq_data)[:, 0]
    eq_sol = qd_to_numpy(info.equalities.sol_params)[:, 0]
    for i_e in range(rigid.n_equalities):
        assert eq_type[i_e] == gs.EQUALITY_TYPE.JOINT, "only joint equalities are ported"
        dof1 = joints_dof_start[eq_obj1[i_e]] - entity.dof_start
        dof2 = joints_dof_start[eq_obj2[i_e]] - entity.dof_start
        equalities.append(np.concatenate([[dof1, dof2], eq_data[i_e, :5], eq_sol[i_e]]))
    model["equalities"] = np.array(equalities).reshape(-1, 14)
    assert np.all(model["frictionloss"] == 0.0), "frictionloss rows are not ported"
    assert np.all(model["qpos0"] == 0.0) or len(dofs) == 6, "a joint offset qpos0 is not ported"
    return model


def far_cloth(shift_y: float = 0.0):
    """A sheet lying on the ground of the scene, far from the rigid bodies."""
    verts, tris = scenes.grid(16, 0.6)
    return verts + FAR_CLOTH_OFFSET + np.array([0.0, shift_y, 0.0]), tris
