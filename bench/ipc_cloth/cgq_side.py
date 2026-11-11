"""Run the cloth scenes in CGQ and save the positions and velocities of every frame (run with CGQ's own venv).

Every scene runs CGQ's contact defaults (adaptive kappa per vertex, CCD partition on). The scenes of the contact between
cloths also dump the contact closure of the first Newton iteration of every frame, which `recompute()` classifies:
their active point-triangle and edge-edge pairs (global vertex stencils and distance flags) are saved per frame.
"""

import argparse
import contextlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cgq", required=True, help="the CGQ checkout, for its benchmarks package")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--resolution", type=int, default=10)
parser.add_argument("--only", nargs="*", default=None)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
parser.add_argument("--tag", default="", help="suffix of the saved runs")
parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
args = parser.parse_args()
sys.path.insert(0, args.cgq)

from benchmarks.bvh_ee_dual_dataset import write_selected_newton_only  # noqa: E402
from qipc import Cloth, Scene, halfplane, trimesh  # noqa: E402
from qipc.offline import ContactClosure  # noqa: E402

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
contact_scenes = scenes.make_contact_scenes(args.resolution)
contact_names = {cloth_scene.name for cloth_scene in contact_scenes}

for cloth_scene in scenes.make_scenes(args.n) + contact_scenes:
    if args.only and cloth_scene.name not in args.only:
        continue
    is_closure = cloth_scene.name in contact_names
    workspace = Path(tempfile.mkdtemp())
    closure_overrides = {}
    if is_closure:
        closure_overrides = {
            "extras/contact_closure/enable": 1,
            "extras/contact_closure/frame_begin": 0,
            "extras/contact_closure/frame_end": cloth_scene.n_frames - 1,
            "extras/contact_closure/max_dumps": 100000,
        }
    scene = Scene(
        workspace=str(workspace),
        dt=scenes.DT,
        gravity=scenes.GRAVITY,
        **{
            "contact/enable": int(cloth_scene.has_ground or len(cloth_scene.cloths) > 1 or is_closure),
            "contact/d_hat": scenes.D_HAT,
            "newton/velocity_tol": args.velocity_tol,
            "newton/max_iter": 1024,
            "line_search/max_iter": 12,
            **closure_overrides,
        },
    )
    for i_cloth, cloth in enumerate(cloth_scene.cloths):
        geo = trimesh(cloth.verts, cloth.tris)
        Cloth().apply_to(
            geo,
            youngs_modulus=scenes.YOUNGS_MODULUS,
            shear_modulus=scenes.SHEAR_MODULUS,
            mass_density=scenes.MASS_DENSITY,
            thickness=scenes.THICKNESS,
            bending=cloth_scene.bending,
            bending_youngs_modulus=scenes.BENDING_YOUNGS_MODULUS,
            is_fixed=cloth.pinned.astype(np.int32),
        )
        scene.geometries.create(f"cloth{i_cloth}", geo)
    if cloth_scene.has_ground:
        scene.geometries.create("ground", halfplane(P=scenes.GROUND_POS, N=scenes.GROUND_NORMAL))
    scene.init()

    fe = scene.finite_element
    positions = [fe.x.cpu().numpy().copy()]
    velocities = [fe.velocities.cpu().numpy().copy()]
    newton_iters, step_ms = [], []
    closure_dir = workspace / "contact_closure"
    with write_selected_newton_only(closure_dir) if is_closure else contextlib.nullcontext():
        for _ in range(cloth_scene.n_frames):
            t = time.perf_counter()
            scene.step()
            step_ms.append(1e3 * (time.perf_counter() - t))
            positions.append(fe.x.cpu().numpy().copy())
            velocities.append(fe.velocities.cpu().numpy().copy())
            newton_iters.append(scene.solver.newton_iters)

    # The active pairs of the first Newton iteration of every frame, as CGQ classifies them
    pairs = {f"{ch}_{name}": [] for ch in ("pt", "ee") for name in ("stencil", "flag", "offsets")}
    pairs["kappa_scale"] = []
    if is_closure:
        closure = ContactClosure(closure_dir)
        for ch in ("pt", "ee"):
            pairs[f"{ch}_offsets"].append(0)
        for frame in range(cloth_scene.n_frames):
            dyn = closure.load_dynamic(frame, 0)
            # The adaptive stiffness of the vertices at the first Newton iteration, the cloth vertices first
            kappa = np.asarray(dyn.get("adaptive_kappa_vert", np.ones(len(cloth_scene.verts)))).reshape(-1)
            pairs["kappa_scale"].append(kappa[: len(cloth_scene.verts)])
            result = scene.contact_system.recompute(dyn["positions"], dyn["trajectory_end_positions"])
            for ch in ("pt", "ee"):
                table = getattr(result, ch)
                active = table.active.cpu().numpy().astype(bool)
                stencil = table.stencil.cpu().numpy().reshape(table.n_candidates, 4)[active]
                pairs[f"{ch}_stencil"].append(stencil)
                pairs[f"{ch}_flag"].append(table.flag.cpu().numpy().astype(np.int32)[active])
                pairs[f"{ch}_offsets"].append(pairs[f"{ch}_offsets"][-1] + len(stencil))
        for ch in ("pt", "ee"):
            pairs[f"{ch}_stencil"] = np.concatenate(pairs[f"{ch}_stencil"]).astype(np.int32)
            pairs[f"{ch}_flag"] = np.concatenate(pairs[f"{ch}_flag"])
            pairs[f"{ch}_offsets"] = np.array(pairs[f"{ch}_offsets"], dtype=np.int64)
    path = out_dir / f"cgq_{cloth_scene.name}{args.tag}.npz"
    np.savez(
        path,
        positions=np.array(positions),
        velocities=np.array(velocities),
        newton_iters=np.array(newton_iters),
        step_ms=np.array(step_ms),
        **({key: np.asarray(value) for key, value in pairs.items()} if is_closure else {}),
    )
    n_active = (
        f", active pairs per frame max pt {np.diff(pairs['pt_offsets']).max()} ee {np.diff(pairs['ee_offsets']).max()}"
        if is_closure
        else ""
    )
    print(
        f"{cloth_scene.name}: {cloth_scene.n_frames} frames, newton iters max {max(newton_iters)}, "
        f"median step {np.median(step_ms):.2f} ms, min y {np.array(positions)[..., 1].min():.6f}{n_active} -> {path}",
        flush=True,
    )
