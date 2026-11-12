"""Run the rigid scenes in CGQ and save the positions and velocities of the cloth at every frame, with the active pairs
of the first Newton iteration of every frame (run with CGQ's own venv, after genesis_side.py --dump-mesh).

The rigid geometry of a scene is the mesh of the Genesis proxies (see scenes.py), a fixed box being CGQ's RigidBody fixed
in the world, and a free box a singleton minimal-coordinate root of the mass properties of `BoxSpec`, whose contact goes
through its reduced-KKT proxy. The pose and the velocity of the root of every free box are saved at every frame. The
contact closure of the first Newton iteration of every frame is classified by `recompute()`, its global vertex indices
mapped to the contact vertex indices of Genesis by the positions of the vertices at the first frame (the cloth, then the
rigid mesh).
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rigid_scenes as scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cgq", required=True, help="the CGQ checkout, for its benchmarks package")
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
parser.add_argument("--line-search-max-iter", type=int, default=12)
parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
args = parser.parse_args()
sys.path.insert(0, args.cgq)

from benchmarks.bvh_ee_dual_dataset import write_selected_newton_only  # noqa: E402
from qipc import Cloth, RigidBody, Scene, halfplane, trimesh  # noqa: E402
from qipc._src.native.solver import RigidBodyDynamics  # noqa: E402
from qipc.offline import ContactClosure  # noqa: E402

out_dir = Path(args.out)
for rigid_scene in scenes.make_scenes(args.n):
    if args.only and rigid_scene.name not in args.only:
        continue
    rigid_mesh = np.load(out_dir / f"rigid_mesh_{rigid_scene.name}.npz")
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
            "line_search/max_iter": args.line_search_max_iter,
            "extras/contact_closure/enable": 1,
            "extras/contact_closure/frame_begin": 0,
            "extras/contact_closure/frame_end": rigid_scene.n_frames - 1,
            "extras/contact_closure/max_dumps": 100000,
        },
    )
    for i_cloth, cloth in enumerate(rigid_scene.cloths):
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
        scene.geometries.create(f"cloth{i_cloth}", geo)
    if rigid_scene.has_ground:
        scene.geometries.create("ground", halfplane(P=scenes.GROUND_POS, N=scenes.GROUND_NORMAL))
    # One box per scene, whose mesh is the whole rigid mesh of Genesis
    (box,) = rigid_scene.boxes
    rigid_geo = trimesh(rigid_mesh["verts"], rigid_mesh["tris"].astype(np.int32))
    if box.is_fixed:
        RigidBody().apply_to(rigid_geo, mass_density=1e3, is_fixed=True)
    else:
        mass, com, inertia = box.mass_properties()
        RigidBody().apply_to(rigid_geo, mass=mass, center_of_mass=com, inertia=inertia, minimal_coordinates=True)
    scene.geometries.create("rigid", rigid_geo)
    scene.init()

    rigid = scene.rigid_body
    native_rigid = scene.sim_systems[RigidBodyDynamics]
    fe = scene.finite_element
    if not box.is_fixed:
        mechanism = int(rigid.host("contact_proxy_pairs")[0, 0])
        rigid.omega[mechanism].copy_(rigid.omega.new_tensor(box.omega))

    def rigid_state():
        """The pose and velocity of the root of the box, then the pose of its proxy."""
        if box.is_fixed:
            return np.zeros(20)
        mechanism, proxy = (int(i) for i in rigid.host("contact_proxy_pairs")[0, :2])
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

    # The contact vertices of Genesis: the cloth, then the rigid mesh
    genesis_verts = np.concatenate((rigid_scene.verts, rigid_mesh["verts"]))
    closure = ContactClosure(closure_dir)
    pairs = {f"{ch}_{name}": [] for ch in ("pt", "ee") for name in ("stencil", "flag", "offsets")}
    for ch in ("pt", "ee"):
        pairs[f"{ch}_offsets"].append(0)
    to_genesis = None
    for frame in range(rigid_scene.n_frames):
        dyn = closure.load_dynamic(frame, 0)
        if to_genesis is None:
            # The global vertices of CGQ end with the anchor point of the ground, which takes no point-triangle or
            # edge-edge pair
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
    path = out_dir / f"cgq_{rigid_scene.name}.npz"
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
        f"{rigid_scene.name}: {rigid_scene.n_frames} frames, newton iters max {max(newton_iters)}, median step "
        f"{np.median(step_ms):.2f} ms, min y {np.array(positions)[..., 1].min():.6f}, active pairs per frame max pt "
        f"{np.diff(pairs['pt_offsets']).max()} ee {np.diff(pairs['ee_offsets']).max()} -> {path}",
        flush=True,
    )
