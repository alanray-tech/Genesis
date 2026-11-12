"""Run the rigid parity case 2 (box_on_cloth, see rigid_scenes.py) in CGQ's Genesis rigid model (rigid_forest/model =
"genesis") and save it as cgq_side.py does, into data_gm/, so that genesis_side.py --data data_gm replays it unchanged
(run with CGQ's own venv).

The scene is cgq_side.py's: the cloth, the ground and the box whose mesh is the proxy mesh of Genesis (copied from data/),
the box a minimal-coordinate root of the mass properties of its BoxSpec. Genesis's model of a free box needs no per-dof
numbers (armature, damping and actuation are zero), only the link frame of the box in its principal frame, which is the
BoxSpec rotation at the first frame.
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import genesis_model_scenes as gms  # noqa: E402
import rigid_scenes as scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cgq", required=True, help="the CGQ checkout, for its benchmarks package")
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
args = parser.parse_args()
sys.path.insert(0, args.cgq)

from benchmarks.bvh_ee_dual_dataset import write_selected_newton_only  # noqa: E402
from qipc import Cloth, RigidBody, Scene, halfplane, trimesh  # noqa: E402
from qipc._src.native.solver import RigidBodyDynamics  # noqa: E402
from qipc.offline import ContactClosure  # noqa: E402

out_dir = gms.DATA
out_dir.mkdir(parents=True, exist_ok=True)
(rigid_scene,) = [s for s in scenes.make_scenes(16) if s.name == "box_on_cloth"]
shutil.copyfile(gms.SOURCE_DATA / "rigid_mesh_box_on_cloth.npz", out_dir / "rigid_mesh_box_on_cloth.npz")
rigid_mesh = np.load(out_dir / "rigid_mesh_box_on_cloth.npz")
workspace = Path(tempfile.mkdtemp())
scene = Scene(
    workspace=str(workspace),
    dt=scenes.DT,
    gravity=scenes.GRAVITY,
    **{
        "contact/enable": 1,
        "contact/d_hat": scenes.D_HAT,
        "newton/velocity_tol": args.velocity_tol,
        "newton/max_iter": 1024,
        "line_search/max_iter": 12,
        "rigid_forest/model": "genesis",
        "extras/contact_closure/enable": 1,
        "extras/contact_closure/frame_begin": 0,
        "extras/contact_closure/frame_end": rigid_scene.n_frames - 1,
        "extras/contact_closure/max_dumps": 100000,
    },
)
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
    is_fixed=cloth.pinned.astype(np.int32),
)
scene.geometries.create("cloth0", geo)
scene.geometries.create("ground", halfplane(P=scenes.GROUND_POS, N=scenes.GROUND_NORMAL))
(box,) = rigid_scene.boxes
mass, com, inertia = box.mass_properties()
rigid_geo = trimesh(rigid_mesh["verts"], rigid_mesh["tris"].astype(np.int32))
RigidBody().apply_to(rigid_geo, mass=mass, center_of_mass=com, inertia=inertia, minimal_coordinates=True)
scene.geometries.create("rigid", rigid_geo)
scene.init()

rigid = scene.rigid_body
native_rigid = scene.sim_systems[RigidBodyDynamics]
fe = scene.finite_element
model = scene.rigid_genesis_model
mechanism, proxy = (int(i) for i in rigid.host("contact_proxy_pairs")[0, :2])
q_body = native_rigid.quat_view_t()[mechanism].cpu().numpy()
link_rotation = gms.quat_mul(q_body * np.array([1.0, -1.0, -1.0, -1.0]), np.array(box.quat))
model.set_root_link_frame(mechanism, link_rotation, np.zeros(3))


def rigid_state():
    """The pose and velocity of the root of the box, then the pose of its proxy (cgq_side.py's layout)."""
    return np.concatenate(
        [
            rigid.t[mechanism].cpu().numpy(),
            rigid.quat[mechanism].cpu().numpy(),
            rigid.v[mechanism].cpu().numpy(),
            rigid.omega[mechanism].cpu().numpy(),
            native_rigid.t_view_t()[proxy].cpu().numpy(),
            native_rigid.quat_view_t()[proxy].cpu().numpy(),
        ]
    )


positions = [fe.x.cpu().numpy().copy()]
velocities = [fe.velocities.cpu().numpy().copy()]
rigid_states = [rigid_state()]
newton_iters, ls_iters, pcg_iters, step_ms = [], [], [], []
closure_dir = workspace / "contact_closure"
with write_selected_newton_only(closure_dir):
    for _ in range(rigid_scene.n_frames):
        t = time.perf_counter()
        scene.step()
        step_ms.append(1e3 * (time.perf_counter() - t))
        positions.append(fe.x.cpu().numpy().copy())
        velocities.append(fe.velocities.cpu().numpy().copy())
        rigid_states.append(rigid_state())
        newton_iters.append(scene.solver.newton_iters)
        ls_iters.append(scene.solver.total_ls_iters)
        pcg_iters.append(scene.solver.total_pcg_iters)

genesis_verts = np.concatenate((rigid_scene.verts, rigid_mesh["verts"]))
closure = ContactClosure(closure_dir)
pairs = {f"{ch}_{name}": [] for ch in ("pt", "ee") for name in ("stencil", "flag", "offsets")}
for ch in ("pt", "ee"):
    pairs[f"{ch}_offsets"].append(0)
to_genesis = None
for frame in range(rigid_scene.n_frames):
    dyn = closure.load_dynamic(frame, 0)
    if to_genesis is None:
        cgq_verts = np.asarray(dyn["positions"]).reshape(-1, 3)[: len(genesis_verts)]
        distances = np.linalg.norm(cgq_verts[:, None, :] - genesis_verts[None, :, :], axis=-1)
        to_genesis = distances.argmin(axis=1)
        assert distances.min(axis=1).max() < 1e-9, "the vertices of CGQ do not match those of Genesis"
        assert len(np.unique(to_genesis)) == len(to_genesis)
    result = scene.contact_system.recompute(dyn["positions"], dyn["trajectory_end_positions"])
    for ch in ("pt", "ee"):
        table = getattr(result, ch)
        active = table.active.cpu().numpy().astype(bool)
        stencil = table.stencil.cpu().numpy().reshape(table.n_candidates, 4)[active]
        pairs[f"{ch}_stencil"].append(to_genesis[stencil])
        pairs[f"{ch}_flag"].append(table.flag.cpu().numpy().astype(np.int32)[active])
        pairs[f"{ch}_offsets"].append(pairs[f"{ch}_offsets"][-1] + len(stencil))
for ch in ("pt", "ee"):
    pairs[f"{ch}_stencil"] = np.concatenate(pairs[f"{ch}_stencil"]).astype(np.int32)
    pairs[f"{ch}_flag"] = np.concatenate(pairs[f"{ch}_flag"])
    pairs[f"{ch}_offsets"] = np.array(pairs[f"{ch}_offsets"], dtype=np.int64)
path = out_dir / "cgq_box_on_cloth.npz"
np.savez(
    path,
    positions=np.array(positions),
    velocities=np.array(velocities),
    rigid_states=np.array(rigid_states),
    newton_iters=np.array(newton_iters),
    ls_iters=np.array(ls_iters),
    pcg_iters=np.array(pcg_iters),
    step_ms=np.array(step_ms),
    **{key: np.asarray(value) for key, value in pairs.items()},
)
print(
    f"box_on_cloth (genesis model): {rigid_scene.n_frames} frames, newton iters max {max(newton_iters)} total "
    f"{sum(newton_iters)}, median step {np.median(step_ms):.2f} ms, rest box y {rigid_states[-1][1]:.6f} -> {path}",
    flush=True,
)
