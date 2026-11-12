"""Run the Franka press of the rigid parity case 3 in CGQ (see franka_scene.py) and save, at the start of every frame, the
positions of every global vertex and the active pairs of the first Newton iteration, with the counts of the Newton solve
(run with CGQ's own venv, after franka_genesis.py --dump).

The scene and the press controller are those of CGQ's examples/rigid_franka_cloth_grasp.py with --press-table, the
Franka loaded from the MJCF franka_genesis.py writes (the proxy meshes of Genesis as its collision meshes), the table and
the riser being the proxy meshes of Genesis. The pairs are saved in CGQ's global vertex indices, which franka_genesis.py
maps.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import franka_scene as fs  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cgq", required=True, help="the CGQ checkout, for its packages")
parser.add_argument("--frames", type=int, default=fs.N_FRAMES)
parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
args = parser.parse_args()
sys.path.insert(0, args.cgq)

from benchmarks.bvh_ee_dual_dataset import write_selected_newton_only  # noqa: E402
from examples._teleop import matrix_to_quat_wxyz  # noqa: E402
from qipc import AffineBodyConstitution, Cloth, Scene, trimesh  # noqa: E402
from qipc._src.native.solver import RigidContactProxySystem  # noqa: E402
from qipc.geometry import ground  # noqa: E402
from qipc.offline import ContactClosure  # noqa: E402
from qipc.tensor_types import download_scalar  # noqa: E402

out_dir = Path(args.out)
workspace = Path(tempfile.mkdtemp())
scene = Scene(
    workspace=str(workspace),
    dt=fs.DT,
    gravity=fs.GRAVITY,
    **{
        "contact/d_hat": fs.D_HAT,
        "contact/enable": 1,
        "newton/velocity_tol": fs.VELOCITY_TOL,
        "linear_system/tol_rate": 1e-5,
        "contact/init_collision_pair_capacity": 20_000,
        "linear_system/abd_preconditioner": "tree",
        "rigid_proxy/globalization": "merit",
        "extras/contact_closure/enable": 1,
        "extras/contact_closure/frame_begin": 0,
        "extras/contact_closure/frame_end": args.frames - 1,
        "extras/contact_closure/max_dumps": 100000,
    },
)
scene.contact_tabular.default_model(friction_rate=fs.FRICTION, resistance=fs.RESISTANCE)
robot = scene.add_mjcf(
    out_dir / "franka_proxies.xml",
    link_type="rigid",
    minimal_coordinates=True,
    kappa=1e8,
    fix_base=True,
    position=fs.BASE_POS,
    rotation=fs.BASE_EULER_DEG,
    enable_controller=True,
)
fixed_meshes = np.load(out_dir / "franka_fixed_meshes.npz")
for name in ("table", "riser"):
    geo = trimesh(fixed_meshes[f"{name}_verts"], fixed_meshes[f"{name}_tris"].astype(np.int32))
    AffineBodyConstitution().apply_to(geo, kappa=1e8, mass_density=1e3, is_fixed=True)
    scene.geometries.create(name, geo)
cloth_verts, cloth_tris = fs.cloth_mesh()
cloth = trimesh(cloth_verts, cloth_tris)
Cloth().apply_to(
    cloth,
    youngs_modulus=fs.YOUNGS_MODULUS,
    shear_modulus=fs.SHEAR_MODULUS,
    mass_density=fs.MASS_DENSITY,
    thickness=fs.CLOTH_THICKNESS,
    bending="hinge",
    bending_youngs_modulus=fs.BENDING_YOUNGS_MODULUS,
)
scene.geometries.create("cloth", cloth)
scene.geometries.create("ground", ground(height=fs.GROUND_H))
scene.init()

arm = robot[[f"joint{i}" for i in range(1, 8)]]
fingers = robot[["finger_joint1", "finger_joint2"]]
initial_qpos = torch.cat([arm.get_dofs_position(), fingers.get_dofs_position()]).cpu().numpy()
end_link = arm.link_names[-1]
q_home = torch.tensor(fs.HOME_QPOS, dtype=torch.float64, device="cuda")
arm.set_dofs_force_range(-fs.ARM_FORCE_RANGE, fs.ARM_FORCE_RANGE)
robot.control_dofs_position(q_home)

# The press target of the example: the fingertip of the home pose taken to the press point, the hand pointing down
ee_pose = arm.forward_kinematics(q_home[:7].unsqueeze(0), link=end_link)[0]
quat_down = matrix_to_quat_wxyz(ee_pose[:3, :3])
rigid = scene.rigid_body
ee_origin, ee_axis = ee_pose[:3, 3], ee_pose[:3, 2]
finger_reach = []
for slot in fingers.links:
    body = int(slot.geometry.meta["rigid_body_offset"].cpu()[0])
    w, x, y, z = rigid.quat[body]
    rotation = torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    )
    world = rigid.x_bar[rigid.body_id == body] @ rotation.T + rigid.t[body]
    finger_reach.append(((world - ee_origin) @ ee_axis).max())
fingertip = torch.stack((ee_origin.new_zeros(()), ee_origin.new_zeros(()), torch.stack(finger_reach).max()))
target_pos = torch.tensor([fs.CLOTH_X, fs.TABLE_TOP - fs.PRESS_DEPTH, 0.0], dtype=torch.float64, device="cuda")


def step_control() -> None:
    qpos = arm.inverse_kinematics(
        link=end_link,
        pos=target_pos,
        quat=quat_down,
        local_point=fingertip,
        init_qpos=arm.get_dofs_position(),
        max_samples=1,
        max_solver_iters=8,
        damping=0.05,
    )
    arm.control_dofs_position(qpos)
    fingers.control_dofs_position(fs.FINGER_OPEN)


proxy_system = scene.sim_systems[RigidContactProxySystem]
newton_iters, ls_iters, pcg_iters, step_ms, surface_residuals = [], [], [], [], []
restoration_entries = 0
closure_dir = workspace / "contact_closure"
with write_selected_newton_only(closure_dir):
    for _ in range(args.frames):
        step_control()
        t = time.perf_counter()
        scene.step()
        step_ms.append(1e3 * (time.perf_counter() - t))
        newton_iters.append(scene.solver.newton_iters)
        ls_iters.append(scene.solver.total_ls_iters)
        pcg_iters.append(scene.solver.total_pcg_iters)
        surface_residuals.append(float(download_scalar(proxy_system.max_surface_residual_view_t().reshape(()))))
        restoration = scene.solver.rigid_proxy_restoration
        if restoration is not None:
            restoration_entries += restoration.entries

closure = ContactClosure(closure_dir)
positions = []
pairs = {f"{ch}_{name}": [] for ch in ("pt", "ee") for name in ("stencil", "flag", "offsets")}
for ch in ("pt", "ee"):
    pairs[f"{ch}_offsets"].append(0)
for frame in range(args.frames):
    dyn = closure.load_dynamic(frame, 0)
    positions.append(np.asarray(dyn["positions"]).reshape(-1, 3))
    result = scene.contact_system.recompute(dyn["positions"], dyn["trajectory_end_positions"])
    for ch in ("pt", "ee"):
        table = getattr(result, ch)
        active = table.active.cpu().numpy().astype(bool)
        pairs[f"{ch}_stencil"].append(table.stencil.cpu().numpy().reshape(table.n_candidates, 4)[active])
        pairs[f"{ch}_flag"].append(table.flag.cpu().numpy().astype(np.int32)[active])
        pairs[f"{ch}_offsets"].append(pairs[f"{ch}_offsets"][-1] + int(active.sum()))
for ch in ("pt", "ee"):
    pairs[f"{ch}_stencil"] = np.concatenate(pairs[f"{ch}_stencil"]).astype(np.int64)
    pairs[f"{ch}_flag"] = np.concatenate(pairs[f"{ch}_flag"])
    pairs[f"{ch}_offsets"] = np.array(pairs[f"{ch}_offsets"], dtype=np.int64)
path = out_dir / "cgq_franka.npz"
np.savez(
    path,
    positions=np.array(positions),
    initial_qpos=initial_qpos,
    newton_iters=np.array(newton_iters),
    ls_iters=np.array(ls_iters),
    pcg_iters=np.array(pcg_iters),
    step_ms=np.array(step_ms),
    surface_residuals=np.array(surface_residuals),
    restoration_entries=restoration_entries,
    **pairs,
)
print(
    f"franka: {args.frames} frames, newton iters max {max(newton_iters)} total {sum(newton_iters)}, ls {sum(ls_iters)}, "
    f"pcg {sum(pcg_iters)}, median step {np.median(step_ms):.2f} ms, surface residual max {max(surface_residuals):.3e}, "
    f"restoration entries {restoration_entries}, active pairs per frame max pt {np.diff(pairs['pt_offsets']).max()} "
    f"ee {np.diff(pairs['ee_offsets']).max()} -> {path}",
    flush=True,
)
