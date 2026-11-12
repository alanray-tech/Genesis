"""Run the rigid parity case 3 (the Franka press of franka_scene.py) in CGQ's Genesis rigid model and save every frame
for genesis_model_case3_genesis.py (run with CGQ's own venv, after genesis_model_case3_genesis.py --prepare).

The scene is franka_cgq.py's (the Franka loaded from the MJCF of franka_genesis.py --dump, the table and the riser fixed
affine bodies on Genesis's proxy meshes, the cloth, the ground), the Franka in rigid_forest/model = "genesis" with the
model Genesis holds (data_gm/case3_setup.npz: dofs_info, joint equality, EPS), set at the home pose and driven by
Genesis's position actuators to the press target of Genesis's press. Per frame the positions of every global vertex
and the active pairs of the first Newton iteration (from the contact closure), the cloth's positions and velocities, the
Franka's qpos and dof velocities in Genesis's order, and the Newton counts are saved.
"""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import franka_scene as fs  # noqa: E402
import genesis_model_scenes as gms  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cgq", required=True, help="the CGQ checkout, for its packages")
parser.add_argument("--frames", type=int, default=fs.N_FRAMES)
parser.add_argument("--velocity-tol", type=float, default=fs.VELOCITY_TOL)
parser.add_argument("--tag", default="")
args = parser.parse_args()
sys.path.insert(0, args.cgq)
import genesis_model_cgq as gmc  # noqa: E402

from benchmarks.bvh_ee_dual_dataset import write_selected_newton_only  # noqa: E402
from qipc import AffineBodyConstitution, Cloth, Scene, trimesh  # noqa: E402
from qipc.geometry import ground  # noqa: E402
from qipc.offline import ContactClosure  # noqa: E402

setup = np.load(gms.DATA / "case3_setup.npz")
shutil.copyfile(gms.SOURCE_DATA / "franka_fixed_meshes.npz", gms.DATA / "franka_fixed_meshes.npz")
workspace = Path(tempfile.mkdtemp())
scene = Scene(
    workspace=str(workspace),
    dt=fs.DT,
    gravity=fs.GRAVITY,
    **{
        "contact/d_hat": fs.D_HAT,
        "contact/enable": 1,
        "newton/velocity_tol": args.velocity_tol,
        "newton/max_iter": 1024,
        "linear_system/tol_rate": 1e-5,
        "contact/init_collision_pair_capacity": 20_000,
        "rigid_proxy/globalization": "merit",
        "rigid_forest/model": "genesis",
        "extras/contact_closure/enable": 1,
        "extras/contact_closure/frame_begin": 0,
        "extras/contact_closure/frame_end": args.frames - 1,
        "extras/contact_closure/max_dumps": 100000,
    },
)
scene.contact_tabular.default_model(friction_rate=fs.FRICTION, resistance=fs.RESISTANCE)
scene.add_mjcf(
    gms.copy_franka_assets(),
    link_type="rigid",
    minimal_coordinates=True,
    fix_base=True,
    position=fs.BASE_POS,
    rotation=fs.BASE_EULER_DEG,
)
fixed_meshes = np.load(gms.DATA / "franka_fixed_meshes.npz")
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

model = scene.rigid_genesis_model
perm = gmc.dof_map(model, [str(j) for j in setup["model_joint_names"]], "franka")
gmc.load_model(scene, setup, perm)
model.set_state(gmc.cgq_qpos(setup["home_qpos"], perm, model.n_qs, "franka"), np.zeros(len(perm)))
model.control_dofs_position(setup["qpos_press"], dofs=perm)
fe = scene.finite_element


def genesis_order(values):
    return np.asarray(values)[perm]


cloth_x, cloth_v, qpos, dofs_vel = [], [], [], []
newton_iters, ls_iters, pcg_iters, step_ms = [], [], [], []
closure_dir = workspace / "contact_closure"
with write_selected_newton_only(closure_dir):
    for _ in range(args.frames):
        cloth_x.append(fe.x.cpu().numpy().copy())
        cloth_v.append(fe.velocities.cpu().numpy().copy())
        qpos.append(genesis_order(model.qpos))
        dofs_vel.append(genesis_order(model.dofs_vel.cpu().numpy()))
        t = time.perf_counter()
        scene.step()
        step_ms.append(1e3 * (time.perf_counter() - t))
        newton_iters.append(scene.solver.newton_iters)
        ls_iters.append(scene.solver.total_ls_iters)
        pcg_iters.append(scene.solver.total_pcg_iters)
        print(f"frame {len(newton_iters) - 1}: newton {newton_iters[-1]} pcg {pcg_iters[-1]}", flush=True)

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
path = gms.DATA / f"cgq_case3{args.tag}.npz"
np.savez(
    path,
    positions=np.array(positions),
    cloth_x=np.array(cloth_x),
    cloth_v=np.array(cloth_v),
    qpos=np.array(qpos),
    dofs_vel=np.array(dofs_vel),
    newton_iters=np.array(newton_iters),
    ls_iters=np.array(ls_iters),
    pcg_iters=np.array(pcg_iters),
    step_ms=np.array(step_ms),
    velocity_tol=np.array(args.velocity_tol),
    **pairs,
)
print(
    f"case 3 (genesis model): {args.frames} frames, newton total {sum(newton_iters)} max {max(newton_iters)}, ls "
    f"{sum(ls_iters)}, pcg {sum(pcg_iters)}, median step {np.median(step_ms):.2f} ms, active pairs per frame max pt "
    f"{np.diff(pairs['pt_offsets']).max()} ee {np.diff(pairs['ee_offsets']).max()}, fingers {qpos[-1][-2:]} -> {path}",
    flush=True,
)
