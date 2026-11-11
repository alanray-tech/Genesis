"""Time the stages of the Newton solve of the cloth on the multilayer scene at the resolution of the reference benchmark.

The scene is multilayer of Genesis-ipc-ref-pr6/examples/newton_coupling/multilayer_cloth_benchmark.py: four horizontal
sheets of 80, 40, 20 and 10 subdivisions (8804 vertices) dropped onto the ground, with its contact parameters (d_hat
1e-3, resistance 1e5, no friction) and membrane (E 6e4, G 3e4), and the default tolerance of the Newton solve. Run it with
QD_KERNEL_PROFILER=1 in an exclusive GPU lease: the GPU time of every kernel is summed per stage over the frames. The
first call of the assembly kernel, timed with a cold compile cache, gives its compile time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

import quadrants as qd
from quadrants.profiler.kernel_profiler import get_default_kernel_profiler

import genesis as gs

# The solver modules read the types of the precision, set once Genesis is initialized
gs.init(backend=gs.gpu, precision="64", logging_level="warning")
import genesis.engine.solvers.newton_solver as newton_solver  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import scenes  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--warmup", type=int, default=30, help="frames before the profile, the sheets landing by then")
parser.add_argument("--data", default=str(Path(__file__).parent / "data"))
parser.add_argument("--results", default=str(Path(__file__).parent / "results"))
args = parser.parse_args()

STAGES = {
    "broad phase": ("kernel_contact_boxes", "build_bvh", "kernel_contact_query", "kernel_contact_filter"),
    "narrow phase + assembly": ("kernel_assemble_rows",),
    "matrix (BCOO sort)": ("bcoo",),
    "CCD + partition": ("kernel_contact_ccd", "kernel_contact_partition"),
    "PCG": ("kernel_pcg_start", "kernel_pcg_loop"),
    "line search": ("kernel_start_line_search", "kernel_line_search_trial"),
    "friction pairs": ("kernel_contact_friction_pairs",),
    "adaptive kappa": ("kernel_contact_kappa_iteration", "kernel_refresh_energy_start"),
}

assembly_compile = {}
kernel_assemble_rows = newton_solver.kernel_assemble_rows


def timed_assemble_rows(*kernel_args):
    """The assembly kernel, its first call timed, compile included."""
    if not assembly_compile:
        qd.sync()
        start = time.perf_counter()
        kernel_assemble_rows(*kernel_args)
        qd.sync()
        assembly_compile["first_call_s"] = time.perf_counter() - start
    else:
        kernel_assemble_rows(*kernel_args)


newton_solver.kernel_assemble_rows = timed_assemble_rows

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=scenes.DT, gravity=scenes.GRAVITY),
    ipc_options=gs.options.IPCOptions(d_hat=1e-3),
    show_viewer=False,
)
scene.add_entity(morph=gs.morphs.Plane(pos=scenes.GROUND_POS, normal=scenes.GROUND_NORMAL))
n_verts = 0
for i_sheet, (n, size, height) in enumerate(((80, 0.5, 0.10), (40, 0.3, 0.14), (20, 0.2, 0.16), (10, 0.1, 0.18))):
    sheet = scenes.sheet(n, n, size, size, (0.0, height, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    path = Path(args.data) / f"timing_multilayer_{i_sheet}.obj"
    path.write_text(trimesh.exchange.obj.export_obj(trimesh.Trimesh(sheet.verts, sheet.tris, process=False), digits=17))
    scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path)),
        material=gs.materials.FEM.Cloth(
            E=6e4,
            shear_modulus=3e4,
            rho=scenes.MASS_DENSITY,
            thickness=scenes.THICKNESS,
            bending="quadratic",
            bending_E=scenes.BENDING_YOUNGS_MODULUS,
            friction_mu=0.0,
            contact_resistance=1e5,
        ),
    )
    n_verts += len(sheet.verts)
scene.build()
newton = scene.sim.newton_solver

# Warm up until the sheets rest on one another, then profile
for _ in range(args.warmup):
    scene.step()
qd.profiler.clear_kernel_profiler_info()
step_ms, n_pairs = [], []
for _ in range(args.frames):
    qd.sync()
    start = time.perf_counter()
    scene.step()
    qd.sync()
    step_ms.append(1e3 * (time.perf_counter() - start))
    n_pairs.append(newton.ipc_contact.n_pairs)
profiler = get_default_kernel_profiler()
profiler._update_records()
totals = {}
for record in profiler._traced_records:
    totals[record.name] = totals.get(record.name, 0.0) + record.kernel_time
stages_ms = {stage: 0.0 for stage in (*STAGES, "other")}
for name, total in totals.items():
    stage = next((stage for stage, keys in STAGES.items() if any(key in name for key in keys)), "other")
    stages_ms[stage] += total / args.frames
result = {
    "n_vertices": n_verts,
    "frames": args.frames,
    "step_ms_median": float(np.median(step_ms)),
    "pairs_max": [int(max(p[k] for p in n_pairs)) for k in range(2)],
    "stages_ms_per_frame": stages_ms,
    "kernels_ms_per_frame": {name: total / args.frames for name, total in sorted(totals.items(), key=lambda x: -x[1])},
    "assembly_first_call_s": assembly_compile.get("first_call_s"),
}
Path(args.results, "stage_timings_multilayer.json").write_text(json.dumps(result, indent=1))
print(json.dumps({key: value for key, value in result.items() if key != "kernels_ms_per_frame"}, indent=1))
