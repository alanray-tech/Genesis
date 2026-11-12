"""Genesis side of the rigid parity case 3 (the Franka press of franka_scene.py) against CGQ's Genesis rigid model, in
double precision.

--prepare builds the scene of franka_genesis.py and saves into data_gm/case3_setup.npz the press target of its press
(the fingertips 2 cm below the table top, through the hand, by Genesis's inverse kinematics from the home pose), the
fingers held open (genesis_model_scenes.FINGER_OPEN_CTRL, Genesis's own fingers command closing them into a native
finger-finger contact), and the model of the Franka (genesis_model_scenes.entity_model); genesis_model_case3_cgq.py
loads it.

Otherwise every frame k of CGQ's run (data_gm/cgq_case3.npz) is replayed for one step: the cloth, the Franka's qpos and
dof velocities of CGQ's frame k are set, the proxies put on their links, the active pairs holding a cloth vertex compared
with CGQ's first Newton iteration (franka_genesis.py's rule), Genesis steps under the same press target, and every
contact vertex (cloth, moving proxies) is compared with CGQ's frame k + 1 against 2 velocity_tol dt, the Newton
iterations, line-search trials and PCG iterations laid beside CGQ's.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir, qd_to_numpy

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ipc_cloth"))

import franka_scene as fs  # noqa: E402
import genesis_model_scenes as gms  # noqa: E402
from contact_probe import compare_pairs, genesis_active_pairs, init_proxies, publish_start_point  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--fast-math", type=int, default=1)
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--velocity-tol", type=float, default=fs.VELOCITY_TOL)
parser.add_argument("--tag", default="")
args = parser.parse_args()
gms.DATA.mkdir(parents=True, exist_ok=True)
gms.RESULTS.mkdir(parents=True, exist_ok=True)

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning", debug=not args.fast_math)
from newton_counters import NewtonCounters, compare_counts  # noqa: E402

cloth_verts, cloth_tris = fs.cloth_mesh()
cloth_path = gms.DATA / "franka_cloth.obj"
cloth_path.write_text(
    trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth_verts, cloth_tris, process=False), digits=17)
)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=fs.DT, gravity=fs.GRAVITY),
    newton_options=gs.options.NewtonOptions(velocity_tol=args.velocity_tol, max_iterations=1024),
    ipc_options=gs.options.IPCOptions(d_hat=fs.D_HAT),
    show_viewer=False,
)
scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, fs.GROUND_H, 0.0), normal=(0.0, 1.0, 0.0)))
rigid_material = gs.materials.Rigid(coup_friction=fs.FRICTION)
scene.add_entity(morph=gs.morphs.Box(size=fs.TABLE_SIZE, pos=fs.TABLE_POS, fixed=True), material=rigid_material)
scene.add_entity(
    morph=gs.morphs.Box(size=(fs.RISER_SIZE,) * 3, pos=fs.RISER_POS, fixed=True), material=rigid_material
)
cloth = scene.add_entity(
    morph=gs.morphs.Mesh(file=str(cloth_path)),
    material=gs.materials.FEM.Cloth(
        E=fs.YOUNGS_MODULUS,
        rho=fs.MASS_DENSITY,
        thickness=fs.CLOTH_THICKNESS,
        shear_modulus=fs.SHEAR_MODULUS,
        bending="hinge",
        bending_E=fs.BENDING_YOUNGS_MODULUS,
        friction_mu=fs.FRICTION,
        contact_resistance=fs.RESISTANCE,
    ),
)
franka_path = Path(get_assets_dir()) / "xml" / "franka_emika_panda" / "panda.xml"
franka = scene.add_entity(
    morph=gs.morphs.MJCF(file=str(franka_path), pos=fs.BASE_POS, quat=fs.BASE_QUAT), material=rigid_material
)
scene.build()
newton = scene.sim.newton_solver
data_manager = newton.data_manager
n_cloth = cloth.n_vertices
proxies_link_idx = qd_to_numpy(data_manager.proxies_info.link_idx)
proxies_is_fixed = qd_to_numpy(data_manager.proxies_info.is_fixed)
proxy_vertices_proxy_idx = qd_to_numpy(data_manager.proxy_vertices_info.proxy_idx)
franka.set_dofs_force_range(-fs.ARM_FORCE_RANGE, fs.ARM_FORCE_RANGE, dofs_idx_local=range(7))


def set_home() -> None:
    franka.set_qpos(fs.HOME_QPOS)
    franka.set_dofs_velocity(np.zeros(franka.n_dofs))
    init_proxies(scene)
    publish_start_point(scene)


if args.prepare:
    set_home()
    hand = franka.get_link("hand")
    fingers = (franka.get_link("left_finger"), franka.get_link("right_finger"))
    hand_pos, hand_quat = hand.get_pos().cpu().numpy(), hand.get_quat().cpu().numpy()
    hand_R = gu.quat_to_R(hand_quat)
    verts = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
    fingers_proxy_idx = [i_p for i_p, i_l in enumerate(proxies_link_idx) if i_l in [f.idx for f in fingers]]
    reach = max(
        float(((verts[n_cloth + np.nonzero(proxy_vertices_proxy_idx == i_p)[0]] - hand_pos) @ hand_R[:, 2]).max())
        for i_p in fingers_proxy_idx
    )
    target_tip = np.array([fs.CLOTH_X, fs.TABLE_TOP - fs.PRESS_DEPTH, 0.0])
    qpos_press = franka.inverse_kinematics(link=hand, pos=target_tip - reach * hand_R[:, 2], quat=hand_quat)
    qpos_press = qpos_press.cpu().numpy()
    qpos_press[-2:] = gms.FINGER_OPEN_CTRL
    model = gms.entity_model(scene, franka)
    path = gms.DATA / "case3_setup.npz"
    np.savez(path, qpos_press=qpos_press, home_qpos=np.array(fs.HOME_QPOS), **{f"model_{k}": v for k, v in model.items()})
    print(f"case 3 press target {np.array2string(qpos_press, precision=4)} -> {path}", flush=True)
    sys.exit(0)

setup = np.load(gms.DATA / "case3_setup.npz")
reference = np.load(gms.DATA / f"cgq_case3{args.tag}.npz")
cgq_positions = reference["positions"]
n_frames = len(reference["newton_iters"])
assert float(reference["velocity_tol"]) == args.velocity_tol, "CGQ ran at another velocity_tol"
bound = 2.0 * args.velocity_tol * fs.DT

# The global vertices of CGQ mapped to the contact vertices of Genesis, -1 for the anchor of the ground
set_home()
genesis_verts = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
# Each side's vertices taken to the nearest of the other's at the home pose: one vertex of link5's three merged collision
# meshes lies 3.5e-9 m from its counterpart, every other one within 1e-15
MATCH_TOLERANCE = 1e-8
distances, nearest = cKDTree(genesis_verts).query(cgq_positions[0])
to_genesis = np.where(distances < MATCH_TOLERANCE, nearest, -1)
distances_back, to_cgq = cKDTree(cgq_positions[0]).query(genesis_verts)
assert distances_back.max() < MATCH_TOLERANCE, "a Genesis contact vertex has no CGQ counterpart"
print(f"vertex map: max distance {distances_back.max():.2e}", flush=True)
moving = np.ones(len(genesis_verts), dtype=bool)
moving[n_cloth:] = ~proxies_is_fixed[proxy_vertices_proxy_idx]
franka.control_dofs_position(setup["qpos_press"])

counters = NewtonCounters(gs.options.NewtonOptions().pcg_max_iterations)
replay_errors, cloth_errors, franka_errors, pairs_rows, counts, step_ms = [], [], [], [], [], []
for k in range(n_frames - 1):
    cloth.set_verts_pos(reference["cloth_x"][k])
    cloth.set_verts_vel(reference["cloth_v"][k])
    franka.set_qpos(reference["qpos"][k])
    franka.set_dofs_velocity(reference["dofs_vel"][k])
    init_proxies(scene)
    publish_start_point(scene)
    genesis_pairs = genesis_active_pairs(scene)
    row = {}
    for ch in ("pt", "ee"):
        offsets = reference[f"{ch}_offsets"]
        stencil = reference[f"{ch}_stencil"][offsets[k] : offsets[k + 1]].reshape(-1, 4)
        flag = reference[f"{ch}_flag"][offsets[k] : offsets[k + 1]]
        mapped = to_genesis[stencil]
        is_kept = (mapped >= 0).all(axis=1) & (mapped < n_cloth).any(axis=1)
        row[ch] = compare_pairs(genesis_pairs[ch], mapped[is_kept], flag[is_kept], ch == "ee")
        row[ch]["n_cgq_rigid_only"] = int((~is_kept).sum())
    pairs_rows.append(row)
    counters.take(scene)
    t = time.perf_counter()
    scene.step()
    step_ms.append(1e3 * (time.perf_counter() - t))
    counts.append(counters.take(scene))
    publish_start_point(scene)
    published = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
    error = np.linalg.norm(published - cgq_positions[k + 1][to_cgq], axis=1)
    replay_errors.append(float(error[moving].max()))
    cloth_errors.append(float(error[:n_cloth].max()))
    franka_errors.append(float(error[n_cloth:][moving[n_cloth:]].max()))

replay_errors = np.array(replay_errors)
totals = {ch: {key: int(sum(row[ch][key] for row in pairs_rows)) for key in pairs_rows[0][ch]} for ch in ("pt", "ee")}
newton_reference = {key: reference[key][: n_frames - 1] for key in ("newton_iters", "ls_iters", "pcg_iters")}
result = {
    "backend": args.backend,
    "fast_math": bool(args.fast_math),
    "bound": bound,
    "replay_errors": replay_errors.tolist(),
    "replay_max_error": float(replay_errors.max()),
    "replay_worst_frame": int(replay_errors.argmax()),
    "replay_frames_above_bound": int((replay_errors > bound).sum()),
    "cloth_replay_errors": cloth_errors,
    "franka_replay_errors": franka_errors,
    "cloth_replay_max_error": float(max(cloth_errors)),
    "franka_replay_max_error": float(max(franka_errors)),
    "step_ms": step_ms,
    "pairs": pairs_rows,
    "pairs_totals": totals,
    "pairs_frames_differing": int(
        sum(any(row[ch]["only_genesis"] or row[ch]["only_cgq"] for ch in ("pt", "ee")) for row in pairs_rows)
    ),
    "pairs_flag_mismatches": int(sum(row[ch]["flag_mismatch"] for row in pairs_rows for ch in ("pt", "ee"))),
    "newton_counts": compare_counts(counts, newton_reference),
    "genesis_step_ms_median": float(np.median(step_ms[1:])),
    "cgq_step_ms_median": float(np.median(reference["step_ms"][1:])),
}
out = gms.RESULTS / f"genesis_model_case3_{args.backend}{args.tag}.json"
out.write_text(json.dumps(result, indent=1))
print(
    f"case 3 [{args.backend}]: replay max {result['replay_max_error']:.3e} (bound {bound:.1e}, "
    f"{result['replay_frames_above_bound']} frames above, worst frame {result['replay_worst_frame']}; cloth max "
    f"{result['cloth_replay_max_error']:.3e}, Franka max {result['franka_replay_max_error']:.3e}), pairs {totals} "
    f"({result['pairs_frames_differing']} frames differ, {result['pairs_flag_mismatches']} flag mismatches), "
    f"newton/ls/pcg totals {result['newton_counts']['totals']} -> {out}",
    flush=True,
)
print(f"PASS: {result['replay_frames_above_bound'] == 0 and result['pairs_frames_differing'] == 0}")
