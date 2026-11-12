"""Profile the steps of the native IPC of Genesis on two parity scenes, to compare with CGQ (cgq_profile.py).

Scenes: cloth_on_box (case 1 of bench/ipc_rigid: a 16 x 16 cloth onto a fixed box, its contact through the rigid contact
proxies) and crossed_drop (bench/ipc_cloth at resolution 10: two sheets of 121 vertices onto the ground), both with the
solver options of their parity runs (velocity_tol 1e-6, max_iterations 1024, line_search_max_iterations 12, fp64).

Every mode runs the trajectory from frame 0 and measures the frames [--start, --end):
- wall: the wall time of every step, synchronized before and after, and its Newton iterations;
- count: per step, the Newton iterations, line-search trials, PCG iterations, the calls of every Quadrants kernel from
  Python and the device-to-host readbacks of 'qd_to_numpy' / 'qd_to_torch' by call site (the readbacks of the counters
  themselves excluded);
- cprofile: cProfile over the window, saved and printed by own time and cumulative time;
- nsys: the window between cudaProfilerStart / Stop, one NVTX range per step and a mark per Newton iteration (run it
  under 'nsys profile --capture-range=cudaProfilerApi');
- compile: the first call of every kernel timed, synchronized before and after (frontend, compile, module load and
  first launch), to run with a fresh compile cache (run.ps1 -Cache <new folder>).
"""

import argparse
import collections
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

import quadrants as qd
from quadrants.lang.kernel import Kernel

import genesis as gs

BENCH = Path(__file__).parent.parent
sys.path.insert(0, str(BENCH / "ipc_cloth"))
sys.path.insert(0, str(BENCH / "ipc_rigid"))

import rigid_scenes  # noqa: E402
import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="gpu")
parser.add_argument("--scene", choices=("cloth_on_box", "crossed_drop"), required=True)
parser.add_argument("--mode", choices=("wall", "count", "cprofile", "nsys", "compile"), required=True)
parser.add_argument("--start", type=int, default=20)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--velocity-tol", type=float, default=scenes.VELOCITY_TOL)
parser.add_argument("--trace-frames", type=int, nargs="*", default=(), help="count mode: trace every Newton iteration")
parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
parser.add_argument("--tag", default="", help="suffix of the output files, to keep the runs of several trees apart")
args = parser.parse_args()
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

gs.init(backend=getattr(gs, args.backend), precision="64", logging_level="warning")
# The solver modules read the types of the precision, set once Genesis is initialized
import genesis.engine.solvers.ipc_contact as ipc_contact  # noqa: E402
import genesis.engine.solvers.newton_solver as newton_solver  # noqa: E402
import genesis.utils.misc as misc  # noqa: E402

materialize_s: dict[str, float] = {}
first_call_s: dict[str, float] = {}
original_materialize = Kernel.materialize
original_kernel_call = Kernel.__call__


def timing_kernel_call(self, *py_args, **kwargs):
    """A kernel call, the first one of each kernel timed with its compile, module load and launch."""
    name = self.func.__name__
    if name in first_call_s:
        return original_kernel_call(self, *py_args, **kwargs)
    qd.sync()
    t = time.perf_counter()
    result = original_kernel_call(self, *py_args, **kwargs)
    qd.sync()
    first_call_s[name] = time.perf_counter() - t
    return result


def timing_materialize(self, key, py_args, arg_features=None):
    """The materialization of a kernel (its frontend, and its compile when not cached), timed per kernel."""
    t = time.perf_counter()
    result = original_materialize(self, key, py_args, arg_features)
    name = self.func.__name__
    materialize_s[name] = materialize_s.get(name, 0.0) + time.perf_counter() - t
    return result


if args.mode == "compile":
    # Every kernel compiles at its first materialization, part of them while the scene builds
    Kernel.materialize = timing_materialize
    Kernel.__call__ = timing_kernel_call

if args.scene == "cloth_on_box":
    scene_spec = rigid_scenes.make_scenes(16)[0]
else:
    scene_spec = scenes.make_contact_scenes(10)[0]
n_frames = scene_spec.n_frames if args.end is None else args.end

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
    newton_options=gs.options.NewtonOptions(
        velocity_tol=args.velocity_tol, max_iterations=1024, line_search_max_iterations=12
    ),
    ipc_options=gs.options.IPCOptions(d_hat=scenes.D_HAT),
    show_viewer=False,
)
if scene_spec.has_ground:
    scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
for i_cloth, cloth in enumerate(scene_spec.cloths):
    mesh_path = data_dir / f"{scene_spec.name}_{i_cloth}.obj"
    mesh_path.write_text(
        trimesh.exchange.obj.export_obj(trimesh.Trimesh(cloth.verts, cloth.tris, process=False), digits=17)
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(mesh_path)),
        material=gs.materials.FEM.Cloth(
            E=scenes.YOUNGS_MODULUS,
            rho=scenes.MASS_DENSITY,
            thickness=scenes.THICKNESS,
            shear_modulus=scenes.SHEAR_MODULUS,
            bending=scene_spec.bending,
            bending_E=scenes.BENDING_YOUNGS_MODULUS,
        ),
    )
if args.scene == "cloth_on_box":
    (box,) = scene_spec.boxes
    scene.add_entity(
        morph=gs.morphs.Box(size=box.size, pos=box.pos, quat=box.quat, fixed=box.is_fixed),
        material=gs.materials.Rigid(rho=box.density, coup_friction=gs.materials.FEM.Cloth().friction_mu),
    )
scene.build()
newton = scene.sim.newton_solver
pcg_max_iterations = int(misc.qd_to_numpy(newton.newton_info.pcg_max_iterations))

# --- instrumentation of the count mode ----------------------------------------------------------------------------
counters = {"kernels": collections.Counter(), "readbacks": collections.Counter(), "ls_trials": 0, "pcg": 0}
is_counting = [False]
is_tracing = [False]
trace_rows = []
last_pcg = [0]
original_readers = {"qd_to_numpy": misc.qd_to_numpy, "qd_to_torch": misc.qd_to_torch}


def counting_kernel_call(self, *py_args, **kwargs):
    if is_counting[0]:
        counters["kernels"][self.func.__name__] += 1
    return original_kernel_call(self, *py_args, **kwargs)


def make_counting_reader(name):
    reader = original_readers[name]

    def counting_reader(*reader_args, **reader_kwargs):
        frame = sys._getframe(1)
        # 'qd_to_numpy' reads through 'qd_to_torch' on GPU, one readback counted once
        if is_counting[0] and frame.f_code.co_name not in original_readers:
            site = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno} {frame.f_code.co_name}"
            counters["readbacks"][site] += 1
        return reader(*reader_args, **reader_kwargs)

    return counting_reader


original_line_search_trial = newton_solver.kernel_line_search_trial
original_pcg_loop = newton_solver.kernel_pcg_loop
original_end_iteration = newton_solver.kernel_end_iteration


def counted_line_search_trial(*kernel_args):
    counters["ls_trials"] += 1
    original_line_search_trial(*kernel_args)


def counted_pcg_loop(*kernel_args):
    original_pcg_loop(*kernel_args)
    is_counting[0] = False
    last_pcg[0] = pcg_max_iterations - int(original_readers["qd_to_numpy"](newton.newton_loops.n_pcg_iterations_left))
    counters["pcg"] += last_pcg[0]
    is_counting[0] = True


def marked_end_iteration(*kernel_args):
    torch.cuda.nvtx.mark("newton_iteration")
    original_end_iteration(*kernel_args)


def traced_end_iteration(*kernel_args):
    """The end of a Newton iteration, its step, line search, convergence measure and CCD bounds recorded first."""
    if is_tracing[0]:
        is_counting[0] = False
        read = original_readers["qd_to_numpy"]
        envs = newton.newton_envs
        bodies_alpha = read(newton.ipc_state.bodies.alpha)[:, 0]
        trace_rows.append(
            {
                "frame": frame,
                "alpha": float(read(envs.alpha)[0]),
                "ls_trials": int(read(envs.n_line_search_iterations)[0]),
                "max_displacement": float(read(envs.max_displacement)[0]),
                "is_converged": bool(read(envs.is_converged)[0]),
                "energy": float(read(envs.energy_start)[0]),
                "pcg_iterations": last_pcg[0],
                "bodies_ccd_alpha": bodies_alpha.tolist(),
                "n_pairs": list(newton.ipc_contact.n_pairs),
            }
        )
        print("trace", json.dumps(trace_rows[-1]), flush=True)
        is_counting[0] = True
    original_end_iteration(*kernel_args)


if args.mode == "count":
    Kernel.__call__ = counting_kernel_call
    for module in [m for name, m in sys.modules.items() if name.startswith("genesis") and m is not None]:
        for name in original_readers:
            if module.__dict__.get(name) is original_readers[name]:
                setattr(module, name, make_counting_reader(name))
    newton_solver.kernel_line_search_trial = counted_line_search_trial
    newton_solver.kernel_pcg_loop = counted_pcg_loop
    newton_solver.kernel_end_iteration = traced_end_iteration
elif args.mode == "nsys":
    newton_solver.kernel_end_iteration = marked_end_iteration
def segments_distance2(p0, p1, q0, q1):
    """The squared distance between the segments p0p1 and q0q1, row by row (clamped closest points)."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = (u * u).sum(-1), (u * v).sum(-1), (v * v).sum(-1)
    d, e = (u * w).sum(-1), (v * w).sum(-1)
    denominator = np.maximum(a * c - b * b, 1e-300)
    s = np.clip((b * e - c * d) / denominator, 0.0, 1.0)
    t = np.clip((b * s + e) / np.maximum(c, 1e-300), 0.0, 1.0)
    s = np.clip((b * t - d) / np.maximum(a, 1e-300), 0.0, 1.0)
    return (((p0 + s[:, None] * u) - (q0 + t[:, None] * v)) ** 2).sum(-1)


def count_mollified_pairs() -> tuple[int, int]:
    """The edge-edge candidates in the contact band at the end of the step, and those among them near-parallel enough
    to take the mollified barrier (see 'func_pair_terms' in ipc_contact.py)."""
    read = original_readers["qd_to_numpy"]
    contact, info = newton.ipc_contact, newton.ipc_info
    n_ee = min(int(read(contact.pairs.counts)[1]), contact.pairs.ee.shape[0])
    pairs = read(contact.pairs.ee)[:n_ee]
    edges = read(info.edges.verts_idx)
    pos = read(newton.ipc_state.vertices.pos)[:, 0]
    rest_length2 = read(info.edges.rest_length2)
    ratio = float(read(info.mollifier_ratio))
    d_hat, thickness = read(info.vertices.d_hat), read(info.vertices.thickness)
    edges_a, edges_b = edges[pairs[:, 1]], edges[pairs[:, 2]]
    p0, p1, q0, q1 = pos[edges_a[:, 0]], pos[edges_a[:, 1]], pos[edges_b[:, 0]], pos[edges_b[:, 1]]
    # The band of a pair: the mean d_hat plus the largest thickness of each side (see func_pair_barrier_parameters)
    band = 0.5 * (d_hat[edges_a[:, 0]] + d_hat[edges_b[:, 0]]) + thickness[edges_a].max(-1) + thickness[edges_b].max(-1)
    is_in_band = segments_distance2(p0, p1, q0, q1) < band**2
    cross_norm2 = (np.cross(p1 - p0, q1 - q0) ** 2).sum(-1)
    threshold = ratio * rest_length2[pairs[:, 1]] * rest_length2[pairs[:, 2]]
    is_mollified = cross_norm2 < threshold
    if args.trace_frames and frame in args.trace_frames:
        relative = cross_norm2 / np.maximum(threshold, 1e-300)
        print(
            f"mollifier check frame {frame}: {n_ee} candidates, {int(is_mollified.sum())} below the threshold, "
            f"{int(is_in_band.sum())} in band, smallest cross/threshold in band "
            f"{relative[is_in_band].min() if is_in_band.any() else float('nan'):.3g}, ratio {ratio:g}",
            flush=True,
        )
    return int(is_in_band.sum()), int((is_in_band & is_mollified).sum()), int(is_mollified.sum())


# --- trajectory ---------------------------------------------------------------------------------------------------
rows = []
profiler = cProfile.Profile()
for frame in range(n_frames):
    is_window = frame >= args.start
    if is_window and frame == args.start:
        qd.sync()
        if args.mode == "nsys":
            torch.cuda.profiler.start()
        elif args.mode == "cprofile":
            profiler.enable()
    if args.mode == "count" and is_window:
        for key in ("ls_trials", "pcg"):
            counters[key] = 0
        counters["kernels"].clear()
        counters["readbacks"].clear()
        is_counting[0] = True
    is_tracing[0] = args.mode == "count" and frame in args.trace_frames
    if args.mode == "nsys" and is_window:
        torch.cuda.nvtx.range_push(f"step {frame}")
    qd.sync()
    t = time.perf_counter()
    scene.step()
    qd.sync()
    step_ms = 1e3 * (time.perf_counter() - t)
    if args.mode == "nsys" and is_window:
        torch.cuda.nvtx.range_pop()
    is_counting[0] = False
    if not is_window:
        continue
    row = {
        "frame": frame,
        "step_ms": step_ms,
        "newton_iterations": int(original_readers["qd_to_numpy"](newton.newton_envs.n_iterations)[0]),
        "n_pairs": list(newton.ipc_contact.n_pairs),
    }
    if args.mode == "count":
        ee_in_band, ee_mollified, ee_parallel = count_mollified_pairs()
        row.update(
            ee_in_band=ee_in_band,
            ee_mollified=ee_mollified,
            ee_parallel=ee_parallel,
            ls_trials=counters["ls_trials"],
            pcg_iterations=counters["pcg"],
            kernel_calls=sum(counters["kernels"].values()),
            kernels=dict(counters["kernels"]),
            readbacks=sum(counters["readbacks"].values()),
            readback_sites=dict(counters["readbacks"]),
        )
    rows.append(row)
    print(
        f"frame {frame}: {step_ms:.1f} ms, {row['newton_iterations']} Newton iterations"
        + (
            f", EE in band {row['ee_in_band']}, mollified {row['ee_mollified']}, near-parallel candidates "
            f"{row['ee_parallel']}"
            if args.mode == "count"
            else ""
        ),
        flush=True,
    )
if args.mode == "nsys":
    qd.sync()
    torch.cuda.profiler.stop()
elif args.mode == "cprofile":
    profiler.disable()
    prof_path = out_dir / f"genesis_{args.scene}_cprofile.prof"
    profiler.dump_stats(prof_path)
    with open(out_dir / f"genesis_{args.scene}_cprofile.txt", "w") as f:
        for key in ("tottime", "cumulative"):
            pstats.Stats(profiler, stream=f).sort_stats(key).print_stats(60)

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
if args.mode == "count":
    for key in ("ls_trials", "pcg_iterations", "kernel_calls", "readbacks"):
        total = int(sum(row[key] for row in rows))
        summary[f"{key}_total"] = total
        summary[f"{key}_per_step"] = total / len(rows)
        summary[f"{key}_per_newton_iteration"] = total / max(iterations.sum(), 1)
    kernels = collections.Counter()
    sites = collections.Counter()
    for row in rows:
        kernels.update(row["kernels"])
        sites.update(row["readback_sites"])
    summary["kernels_total"] = dict(kernels.most_common())
    summary["readback_sites_total"] = dict(sites.most_common())
if args.mode == "compile":
    compile_s = {name: materialize_s.get(name, 0.0) + first_call_s.get(name, 0.0) for name in {*materialize_s, *first_call_s}}
    summary["compile_s"] = dict(sorted(compile_s.items(), key=lambda item: -item[1]))
    summary["compile_s_total"] = sum(compile_s.values())
tag = f"_tol{args.velocity_tol:g}" if args.velocity_tol != scenes.VELOCITY_TOL else ""
tag += "" if args.backend == "gpu" else f"_{args.backend}"
tag += args.tag
(out_dir / f"genesis_{args.scene}_{args.mode}{tag}.json").write_text(
    json.dumps({"summary": summary, "rows": rows, "trace": trace_rows}, indent=1)
)
print(json.dumps(summary, indent=1), flush=True)
