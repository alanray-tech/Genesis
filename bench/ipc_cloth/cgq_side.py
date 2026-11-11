"""Run the cloth scenes in CGQ and save the positions and velocities of every frame (run with CGQ's own venv)."""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from qipc import Cloth, Scene, halfplane, trimesh  # noqa: E402

import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
args = parser.parse_args()
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

for cloth_scene in scenes.make_scenes(args.n):
    scene = Scene(
        workspace=tempfile.mkdtemp(),
        dt=scenes.DT,
        gravity=scenes.GRAVITY,
        **{
            "contact/enable": int(cloth_scene.has_ground),
            "contact/d_hat": scenes.D_HAT,
            "contact/adaptive_kappa_mode": "off",
            "newton/velocity_tol": scenes.VELOCITY_TOL,
            "newton/max_iter": 1024,
            "line_search/max_iter": 12,
        },
    )
    geo = trimesh(cloth_scene.verts, cloth_scene.tris)
    Cloth().apply_to(
        geo,
        youngs_modulus=scenes.YOUNGS_MODULUS,
        shear_modulus=scenes.SHEAR_MODULUS,
        mass_density=scenes.MASS_DENSITY,
        thickness=scenes.THICKNESS,
        bending=cloth_scene.bending,
        bending_youngs_modulus=scenes.BENDING_YOUNGS_MODULUS,
        is_fixed=cloth_scene.pinned.astype(np.int32),
    )
    scene.geometries.create("cloth", geo)
    if cloth_scene.has_ground:
        scene.geometries.create("ground", halfplane(P=scenes.GROUND_POS, N=scenes.GROUND_NORMAL))
    scene.init()

    fe = scene.finite_element
    positions = [fe.x.cpu().numpy().copy()]
    velocities = [fe.velocities.cpu().numpy().copy()]
    newton_iters, step_ms = [], []
    for _ in range(cloth_scene.n_frames):
        t = time.perf_counter()
        scene.step()
        step_ms.append(1e3 * (time.perf_counter() - t))
        positions.append(fe.x.cpu().numpy().copy())
        velocities.append(fe.velocities.cpu().numpy().copy())
        newton_iters.append(scene.solver.newton_iters)
    path = out_dir / f"cgq_{cloth_scene.name}_n{args.n}.npz"
    np.savez(
        path,
        positions=np.array(positions),
        velocities=np.array(velocities),
        newton_iters=np.array(newton_iters),
        step_ms=np.array(step_ms),
        verts=cloth_scene.verts,
        tris=cloth_scene.tris,
        pinned=cloth_scene.pinned,
    )
    print(f"{cloth_scene.name}: {cloth_scene.n_frames} frames, newton iters max {max(newton_iters)}, "
          f"median step {np.median(step_ms):.2f} ms, min y {np.array(positions)[..., 1].min():.6f} -> {path}")
