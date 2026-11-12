"""Profile the steps of CGQ on the two scenes of genesis_profile.py, with CGQ's own venv.

The scenes are those of bench/ipc_rigid/cgq_side.py (cloth_on_box, the rigid mesh dumped by its Genesis side) and
bench/ipc_cloth/cgq_side.py (crossed_drop), without their contact-closure dumps, which write files every frame.

Modes, over the frames [--start, --end) of the trajectory from frame 0:
- wall: the wall time of every step, synchronized before and after, with the Newton iterations, the line-search trials
  and the PCG iterations of the step, read after it;
- nsys: the window between cudaProfilerStart / Stop, one NVTX range per step (run it under 'nsys profile
  --capture-range=cudaProfilerApi').
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

BENCH = Path(__file__).parent.parent
sys.path.insert(0, str(BENCH / "ipc_cloth"))
sys.path.insert(0, str(BENCH / "ipc_rigid"))

import rigid_scenes  # noqa: E402
import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--cgq", default=r"C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-p2a-aeb58956")
parser.add_argument("--scene", choices=("cloth_on_box", "crossed_drop"), required=True)
parser.add_argument("--mode", choices=("wall", "nsys"), required=True)
parser.add_argument("--start", type=int, default=20)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()
sys.path.insert(0, args.cgq)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

from qipc import Cloth, RigidBody, Scene, halfplane, trimesh  # noqa: E402

if args.scene == "cloth_on_box":
    scene_spec = rigid_scenes.make_scenes(16)[0]
else:
    scene_spec = scenes.make_contact_scenes(10)[0]
n_frames = scene_spec.n_frames if args.end is None else args.end

scene = Scene(
    workspace=tempfile.mkdtemp(),
    dt=scenes.DT,
    gravity=scenes.GRAVITY,
    **{
        "contact/enable": 1,
        "contact/d_hat": scenes.D_HAT,
        "newton/velocity_tol": args.velocity_tol,
        "newton/max_iter": 1024,
        "line_search/max_iter": 12,
    },
)
for i_cloth, cloth in enumerate(scene_spec.cloths):
    geo = trimesh(cloth.verts, cloth.tris)
    Cloth().apply_to(
        geo,
        youngs_modulus=scenes.YOUNGS_MODULUS,
        shear_modulus=scenes.SHEAR_MODULUS,
        mass_density=scenes.MASS_DENSITY,
        thickness=scenes.THICKNESS,
        bending=scene_spec.bending,
        bending_youngs_modulus=scenes.BENDING_YOUNGS_MODULUS,
        is_fixed=cloth.pinned.astype(np.int32),
    )
    scene.geometries.create(f"cloth{i_cloth}", geo)
if scene_spec.has_ground:
    scene.geometries.create("ground", halfplane(P=scenes.GROUND_POS, N=scenes.GROUND_NORMAL))
if args.scene == "cloth_on_box":
    rigid_mesh = np.load(BENCH / "ipc_rigid" / "data" / f"rigid_mesh_{scene_spec.name}.npz")
    rigid_geo = trimesh(rigid_mesh["verts"], rigid_mesh["tris"].astype(np.int32))
    RigidBody().apply_to(rigid_geo, mass_density=1e3, is_fixed=True)
    scene.geometries.create("rigid", rigid_geo)
scene.init()
solver = scene.solver

rows = []
for frame in range(n_frames):
    is_window = frame >= args.start
    if is_window and frame == args.start and args.mode == "nsys":
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
    if args.mode == "nsys" and is_window:
        torch.cuda.nvtx.range_push(f"step {frame}")
    torch.cuda.synchronize()
    t = time.perf_counter()
    scene.step()
    torch.cuda.synchronize()
    step_ms = 1e3 * (time.perf_counter() - t)
    if args.mode == "nsys" and is_window:
        torch.cuda.nvtx.range_pop()
    if not is_window:
        continue
    rows.append(
        {
            "frame": frame,
            "step_ms": step_ms,
            "newton_iterations": solver.newton_iters,
            "ls_trials": solver.total_ls_iters,
            "pcg_iterations": solver.total_pcg_iters,
            "min_ccd_alpha": solver.min_ccd_alpha,
            "avg_ccd_alpha": solver.avg_ccd_alpha,
        }
    )
    print(f"frame {frame}: {step_ms:.1f} ms, {rows[-1]['newton_iterations']} Newton iterations", flush=True)
if args.mode == "nsys":
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

steps_ms = np.array([row["step_ms"] for row in rows])
iterations = np.array([row["newton_iterations"] for row in rows])
summary = {
    "scene": args.scene,
    "mode": args.mode,
    "window": [args.start, n_frames],
    "step_ms_median": float(np.median(steps_ms)),
    "step_ms_mean": float(steps_ms.mean()),
    "step_ms_total": float(steps_ms.sum()),
    "newton_iterations_total": int(iterations.sum()),
    "newton_iterations_median": float(np.median(iterations)),
    "ms_per_newton_iteration": float(steps_ms.sum() / max(iterations.sum(), 1)),
}
for key in ("ls_trials", "pcg_iterations"):
    total = int(sum(row[key] for row in rows))
    summary[f"{key}_total"] = total
    summary[f"{key}_per_step"] = total / len(rows)
    summary[f"{key}_per_newton_iteration"] = total / max(iterations.sum(), 1)
tag = f"_tol{args.velocity_tol:g}" if args.velocity_tol != scenes.VELOCITY_TOL else ""
(out_dir / f"cgq_{args.scene}_{args.mode}{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary, indent=1), flush=True)
