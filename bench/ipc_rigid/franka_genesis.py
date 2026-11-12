"""Genesis side of the rigid parity case 3 with CGQ aeb58956, in double precision (see franka_scene.py).

--dump builds the scene and writes the meshes of its rigid contact proxies, which franka_cgq.py loads: every link of the
Franka in the frame of its link, as the collision mesh of its body in a copy of the MJCF of Genesis stripped of its own
geoms, the table and the riser in the world.

Otherwise, at every frame of CGQ's run, the cloth and the proxies of Genesis are set at CGQ's state, the pose of every
proxy fitted to the vertices of its link (the fit error is reported), and the active pairs of the first Newton iteration
holding a cloth vertex compared with CGQ's (the pairs between rigid meshes go through the native rigid contact in
Genesis). The global vertices of CGQ are mapped to the contact vertices of Genesis by their positions at the first frame,
the Franka of Genesis set at CGQ's initial joint positions. Then Genesis runs the press itself, with its own controller:
every frame is checked for an edge crossing a triangle and for the surface residual of every moving proxy at the committed
state against the solve tolerance, and its Newton iterations, line-search trials and PCG iterations are laid beside CGQ's.
"""

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
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
from contact_probe import compare_pairs, genesis_active_pairs, init_proxies, publish_start_point  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--fast-math", type=int, default=1)
parser.add_argument("--dump", action="store_true")
parser.add_argument("--tag", default="")
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
data_dir, results_dir = Path(args.data), Path(args.results)
data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning", debug=not args.fast_math)
from newton_counters import NewtonCounters, compare_counts  # noqa: E402

cloth_verts, cloth_tris = fs.cloth_mesh()
cloth_path = data_dir / "franka_cloth.obj"
cloth_path.write_text(
    trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth_verts, cloth_tris, process=False), digits=17)
)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=fs.DT, gravity=fs.GRAVITY),
    newton_options=gs.options.NewtonOptions(velocity_tol=fs.VELOCITY_TOL),
    ipc_options=gs.options.IPCOptions(d_hat=fs.D_HAT),
    show_viewer=False,
)
scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, fs.GROUND_H, 0.0), normal=(0.0, 1.0, 0.0)))
rigid_material = gs.materials.Rigid(coup_friction=fs.FRICTION)
table = scene.add_entity(morph=gs.morphs.Box(size=fs.TABLE_SIZE, pos=fs.TABLE_POS, fixed=True), material=rigid_material)
riser = scene.add_entity(
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
newton, rigid = scene.sim.newton_solver, scene.sim.rigid_solver
data_manager = newton.data_manager
n_cloth = cloth.n_vertices
proxies_link_idx = qd_to_numpy(data_manager.proxies_info.link_idx)
proxies_is_fixed = qd_to_numpy(data_manager.proxies_info.is_fixed)
proxy_vertices_proxy_idx = qd_to_numpy(data_manager.proxy_vertices_info.proxy_idx)
proxy_vertices_local_pos = qd_to_numpy(data_manager.proxy_vertices_info.local_pos)
links_by_idx = {link.idx: link for entity in rigid.entities for link in entity.links}


def set_home(qpos) -> None:
    franka.set_qpos(qpos)
    init_proxies(scene)
    publish_start_point(scene)


if args.dump:
    set_home(fs.HOME_QPOS)
    verts_world = qd_to_numpy(newton.ipc_state.vertices.pos)[n_cloth:, 0]
    tris = qd_to_numpy(newton.ipc_info.triangles.verts_idx)
    tris = tris[tris.min(axis=1) >= n_cloth] - n_cloth
    mjcf = ET.parse(franka_path).getroot()
    asset = mjcf.find("asset")
    for mesh in asset.findall("mesh"):
        asset.remove(mesh)
    bodies = {body.get("name"): body for body in mjcf.iter("body")}
    for body in bodies.values():
        for geom in body.findall("geom"):
            body.remove(geom)
    fixed_meshes = {}
    for i_p, i_l in enumerate(proxies_link_idx):
        verts_idx = np.nonzero(proxy_vertices_proxy_idx == i_p)[0]
        local_idx = np.full(len(proxy_vertices_proxy_idx), -1)
        local_idx[verts_idx] = np.arange(len(verts_idx))
        proxy_tris = local_idx[tris[np.isin(tris[:, 0], verts_idx)]]
        link = links_by_idx[i_l]
        if link.entity is franka:
            mesh_path = data_dir / f"franka_proxy_{link.name}.obj"
            mesh = trimesh.Trimesh(proxy_vertices_local_pos[verts_idx], proxy_tris, process=False)
            mesh_path.write_text(trimesh.exchange.obj.export_obj(mesh, digits=17))
            ET.SubElement(asset, "mesh", name=f"proxy_{link.name}", file=str(mesh_path))
            ET.SubElement(bodies[link.name], "geom", {"class": "collision", "mesh": f"proxy_{link.name}"})
        else:
            name = "table" if link.entity is table else "riser"
            fixed_meshes[f"{name}_verts"] = verts_world[verts_idx]
            fixed_meshes[f"{name}_tris"] = proxy_tris
        print(f"proxy {i_p}: link {link.name}, {len(verts_idx)} vertices, {len(proxy_tris)} triangles", flush=True)
    (data_dir / "franka_proxies.xml").write_text(ET.tostring(mjcf, encoding="unicode"))
    np.savez(data_dir / "franka_fixed_meshes.npz", **fixed_meshes)
    sys.exit(0)

reference = np.load(data_dir / "cgq_franka.npz")
cgq_positions = reference["positions"]
n_frames = len(cgq_positions)

# The contact vertices of Genesis mapped to the global vertices of CGQ, whose loader merges the vertices of the convex
# pieces of a link closer than 1e-8 (trimesh's merge tolerance), which Genesis keeps apart: the pairs are compared in the
# indices of CGQ
set_home(reference["initial_qpos"])
genesis_verts = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
distances, to_cgq = cKDTree(cgq_positions[0]).query(genesis_verts)
assert distances.max() < 1e-8, "unmatched vertices"
is_cgq_cloth = np.zeros(len(cgq_positions[0]), dtype=bool)
is_cgq_cloth[to_cgq[:n_cloth]] = True
n_merged = len(genesis_verts) - len(np.unique(to_cgq))
print(
    f"vertex map: {len(genesis_verts)} contact vertices, {n_merged} merged by CGQ, max distance {distances.max():.2e}"
)

proxies_state = data_manager.proxies_state
inertial_pos = qd_to_numpy(proxies_state.inertial_pos)[:, 0]
inertial_R = np.stack([gu.quat_to_R(quat) for quat in qd_to_numpy(proxies_state.inertial_quat)[:, 0]])
proxies_verts_idx = [n_cloth + np.nonzero(proxy_vertices_proxy_idx == i_p)[0] for i_p in range(len(proxies_link_idx))]


def set_cgq_state(positions) -> float:
    """Set the cloth and the moving proxies at the positions of CGQ's global vertices, each link pose fitted to the
    vertices of its proxy, and return the largest distance of a published contact vertex from CGQ's."""
    cloth.set_verts_pos(positions[to_cgq[:n_cloth]])
    proxies_pos, proxies_quat = qd_to_numpy(proxies_state.pos), qd_to_numpy(proxies_state.quat)
    for i_p, verts_idx in enumerate(proxies_verts_idx):
        if proxies_is_fixed[i_p]:
            continue
        local = proxy_vertices_local_pos[verts_idx - n_cloth]
        world = positions[to_cgq[verts_idx]]
        local_mean, world_mean = local.mean(axis=0), world.mean(axis=0)
        u, _, vt = np.linalg.svd((world - world_mean).T @ (local - local_mean))
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        translation = world_mean - rotation @ local_mean
        proxies_pos[i_p, 0] = translation + rotation @ inertial_pos[i_p]
        proxies_quat[i_p, 0] = gu.R_to_quat(rotation @ inertial_R[i_p])
    proxies_state.pos.from_numpy(proxies_pos)
    proxies_state.quat.from_numpy(proxies_quat)
    publish_start_point(scene)
    published = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
    return float(np.abs(published - positions[to_cgq]).max())


fit_errors, pairs_rows = [], []
for k in range(n_frames):
    fit_errors.append(set_cgq_state(cgq_positions[k]))
    genesis_pairs = genesis_active_pairs(scene)
    row = {}
    for ch in ("pt", "ee"):
        offsets = reference[f"{ch}_offsets"]
        stencil = reference[f"{ch}_stencil"][offsets[k] : offsets[k + 1]].reshape(-1, 4)
        flag = reference[f"{ch}_flag"][offsets[k] : offsets[k + 1]]
        # The pairs of CGQ between two rigid meshes go through the native rigid contact in Genesis
        is_kept = is_cgq_cloth[stencil].any(axis=1)
        genesis_stencil, genesis_flag = genesis_pairs[ch]
        row[ch] = compare_pairs((to_cgq[genesis_stencil], genesis_flag), stencil[is_kept], flag[is_kept], ch == "ee")
        row[ch]["n_cgq_rigid_only"] = int((~is_kept).sum())
    pairs_rows.append(row)

# Genesis's own press: the fingertips commanded to the target of CGQ's example, through the hand
set_home(fs.HOME_QPOS)
hand = franka.get_link("hand")
fingers = (franka.get_link("left_finger"), franka.get_link("right_finger"))
hand_pos, hand_quat = hand.get_pos().cpu().numpy(), hand.get_quat().cpu().numpy()
hand_R = gu.quat_to_R(hand_quat)
genesis_verts = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
fingers_proxy_idx = [i_p for i_p, i_l in enumerate(proxies_link_idx) if i_l in [finger.idx for finger in fingers]]
reach = max(((genesis_verts[proxies_verts_idx[i_p]] - hand_pos) @ hand_R[:, 2]).max() for i_p in fingers_proxy_idx)
target_tip = np.array([fs.CLOTH_X, fs.TABLE_TOP - fs.PRESS_DEPTH, 0.0])
qpos_press = franka.inverse_kinematics(link=hand, pos=target_tip - reach * hand_R[:, 2], quat=hand_quat)
qpos_press[-2:] = fs.FINGER_OPEN
franka.set_dofs_force_range(-fs.ARM_FORCE_RANGE, fs.ARM_FORCE_RANGE, dofs_idx_local=range(7))
franka.control_dofs_position(qpos_press)

counters = NewtonCounters(gs.options.NewtonOptions().pcg_max_iterations)
counts, residuals, n_intersections, step_ms, tips_y, cloth_min_y = [], [], [], [], [], []
moving = np.nonzero(~proxies_is_fixed)[0]
for k in range(n_frames):
    counters.take(scene)
    t = time.perf_counter()
    scene.step()
    step_ms.append(1e3 * (time.perf_counter() - t))
    counts.append(counters.take(scene))
    links_pos = rigid.get_links_pos().cpu().numpy()
    links_quat = rigid.get_links_quat().cpu().numpy()
    proxies_pos, proxies_quat = qd_to_numpy(proxies_state.pos)[:, 0], qd_to_numpy(proxies_state.quat)[:, 0]
    surface_radius = qd_to_numpy(proxies_state.surface_radius)[:, 0]
    frame_residual = 0.0
    for i_p in moving:
        i_l = proxies_link_idx[i_p]
        link_R = gu.quat_to_R(links_quat[i_l])
        expected_pos = links_pos[i_l] + link_R @ inertial_pos[i_p]
        relative = (link_R @ inertial_R[i_p]).T @ gu.quat_to_R(proxies_quat[i_p])
        angle = np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0))
        residual = np.linalg.norm(proxies_pos[i_p] - expected_pos) + surface_radius[i_p] * angle
        frame_residual = max(frame_residual, float(residual))
    residuals.append(frame_residual)
    publish_start_point(scene)
    n_intersections.append(len(newton.ipc_contact.find_intersections(newton.ipc_state, newton.ipc_info)))
    verts = qd_to_numpy(newton.ipc_state.vertices.pos)[:, 0]
    tips_y.append(float(min(verts[proxies_verts_idx[i_p]][:, 1].min() for i_p in fingers_proxy_idx)))
    cloth_min_y.append(float(verts[:n_cloth, 1].min()))

solve_tolerance = min(fs.VELOCITY_TOL * fs.DT, 0.01 * fs.D_HAT)
totals = {ch: {key: int(sum(row[ch][key] for row in pairs_rows)) for key in pairs_rows[0][ch]} for ch in ("pt", "ee")}
result = {
    "backend": args.backend,
    "fast_math": bool(args.fast_math),
    "n_frames": n_frames,
    "fit_max_error": float(max(fit_errors)),
    "pairs": pairs_rows,
    "pairs_totals": totals,
    "pairs_frames_differing": int(
        sum(any(row[ch]["only_genesis"] or row[ch]["only_cgq"] for ch in ("pt", "ee")) for row in pairs_rows)
    ),
    "pairs_flag_mismatches": int(sum(row[ch]["flag_mismatch"] for row in pairs_rows for ch in ("pt", "ee"))),
    "trajectory_intersections": int(sum(n_intersections)),
    "commit_residuals": residuals,
    "commit_residual_max": float(max(residuals)),
    "solve_tolerance": solve_tolerance,
    "commit_frames_above_tolerance": int((np.array(residuals) > solve_tolerance).sum()),
    "fingertips_min_y": tips_y,
    "cloth_min_y": cloth_min_y,
    "table_top": fs.TABLE_TOP,
    "newton_counts": compare_counts(counts, reference),
    "genesis_step_ms_median": float(np.median(step_ms[1:])),
    "cgq_step_ms_median": float(np.median(reference["step_ms"][1:])),
}
out = results_dir / f"franka_genesis_{args.backend}{args.tag}.json"
out.write_text(json.dumps(result, indent=1))
print(
    f"franka [{args.backend}]: fit max {result['fit_max_error']:.2e}, pairs {totals} "
    f"({result['pairs_frames_differing']} frames differ, {result['pairs_flag_mismatches']} flag mismatches), "
    f"intersections {result['trajectory_intersections']}, commit residual max {result['commit_residual_max']:.2e} "
    f"(tolerance {solve_tolerance:.1e}, {result['commit_frames_above_tolerance']} frames above), fingertips min y "
    f"{min(tips_y):.5f}, cloth min y {min(cloth_min_y):.5f} (table top {fs.TABLE_TOP}), newton/ls/pcg totals "
    f"{result['newton_counts']['totals']}, {len(result['newton_counts']['flagged_frames'])} frames above 1.5x CGQ's "
    f"Newton iterations, step {result['genesis_step_ms_median']:.1f} ms vs CGQ {result['cgq_step_ms_median']:.1f} ms "
    f"-> {out}",
    flush=True,
)
