"""Pair the kernels of Genesis and CGQ by stage of the Newton solve, and print the kernel time per step of each stage on
both sides, from the nsys SQLite exports of the same scene and window (genesis_<scene>.sqlite, cgq_<scene>.sqlite).

Genesis kernels map by name, and the tasks of the kernels that hold several stages by their index (the order of their
top-level loops, see 'func_contact_assemble' and 'func_contact_energy' in ipc_contact.py). CGQ kernels map by name.

    python stage_table.py crossed_drop
"""

import argparse
import collections
import re
import sqlite3
from pathlib import Path

STAGES = (
    "contact assembly (narrow phase + barrier/friction Hessians)",
    "elastic assembly",
    "contact energy",
    "elastic energy + step forward",
    "substep start (energy, init)",
    "PCG",
    "BCOO sort-reduce",
    "broad phase",
    "CCD",
    "friction pairs",
    "kappa",
    "line search / Newton control",
    "other",
)
# The tasks of the pair loops (point-triangle, edge-edge, friction point-triangle, friction edge-edge), after the
# half-plane loop of a scene with a ground and before the rigid-vertex loop of a scene with proxies
CONTACT_TASKS = {
    "crossed_drop": {"assemble_rows": {9, 11, 13, 15, 17}, "line_search_trial": {6, 8, 10, 12, 14}},
    "cloth_on_box": {"assemble_rows": {11, 13, 15, 17, 19}, "line_search_trial": {7, 9, 11, 13}},
}
GENESIS_BY_NAME = (
    (r"^kernel_contact_(boxes|query|filter)|^build_bvh|^kernel_ef_", "broad phase"),
    (r"^kernel_contact_(ccd|partition)|^kernel_scale_directions", "CCD"),
    (r"^kernel_pcg_", "PCG"),
    (r"^kernel_(compose_triplets_key|radix_sort|mark_entry_ends|exclusive_sum|reduce_triplets|set_segments)", "BCOO sort-reduce"),
    (r"^kernel_contact_friction_pairs", "friction pairs"),
    (r"^kernel_(contact_kappa_iteration|refresh_energy_start)", "kappa"),
    (r"^kernel_(start_line_search|end_iteration|flag_failures)", "line search / Newton control"),
    (r"^kernel_init_substep", "substep start (energy, init)"),
)
CGQ_BY_NAME = (
    (r"^cipc_.*assemble|^cipc_count_active", "contact assembly (narrow phase + barrier/friction Hessians)"),
    (r"^(strain_limit_shell_gradient_hessian|quadratic_bending_assemble|fem_kinetic_gradient_hessian|distribute_fem_)", "elastic assembly"),
    (r"^cipc_.*energy|^reduce_(simplex_barrier|friction)_energy", "contact energy"),
    (r"^(strain_limit_shell_energy|quadratic_bending_energy|fem_kinetic_energy|fem_energy_final|fem_step_forward)", "elastic energy + step forward"),
    (r"^(masked_pcg|mas_|pcg_bound)", "PCG"),
    (r"^(drs_|des_|run_segmented_reduce|fast_segmented_reduce|DeviceScan|packed_layout_scan)", "BCOO sort-reduce"),
    (r"^(ef_|bvh_|halfplane_query)", "broad phase"),
    (r"^(ccd_|halfplane_ccd|component_|pre_ccd)", "CCD"),
    (r"kappa", "kappa"),
    (r"^(pipeline_|checkpoint_)", "line search / Newton control"),
)

parser = argparse.ArgumentParser()
parser.add_argument("scene")
parser.add_argument("--genesis", default=None, help="the Genesis export, results/genesis_<scene>.sqlite by default")
parser.add_argument("--cgq", default=None, help="the CGQ export, results/cgq_<scene>.sqlite by default")
parser.add_argument("--frames", type=int, nargs="*", default=None, help="keep only these steps of the window")
args = parser.parse_args()
results = Path(__file__).parent / "results"


def genesis_stage(name: str) -> str:
    match = re.match(r"(kernel_(assemble_rows|line_search_trial))_c\d+_\d+_kernel_(\d+)_", name)
    if match:
        is_contact = int(match.group(3)) in CONTACT_TASKS[args.scene][match.group(2)]
        if match.group(2) == "assemble_rows":
            return STAGES[0] if is_contact else "elastic assembly"
        return "contact energy" if is_contact else "elastic energy + step forward"
    for pattern, stage in GENESIS_BY_NAME:
        if re.search(pattern, name):
            return stage
    return "other"


def cgq_stage(name: str) -> str:
    for pattern, stage in CGQ_BY_NAME:
        if re.search(pattern, name):
            return stage
    return "other"


def stage_times(path: Path, classify, name_column: str) -> tuple[dict, dict, int]:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    steps = [
        (start, end)
        for start, end, text in db.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'step %' ORDER BY start")
        if args.frames is None or int(text.split()[1]) in args.frames
    ]
    times, kernels = collections.Counter(), collections.defaultdict(collections.Counter)
    for k_start, k_end, name_id in db.execute(f"SELECT start, end, {name_column} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if any(start <= k_start <= end for start, end in steps):
            name = strings.get(name_id, str(name_id))
            stage = classify(name)
            times[stage] += k_end - k_start
            kernels[stage][name] += k_end - k_start
    return times, kernels, len(steps)


genesis_path = Path(args.genesis) if args.genesis else results / f"genesis_{args.scene}.sqlite"
# The short names of nsys truncate the Quadrants task names, which carry the task index, and the demangled names of
# CGQ carry its namespace and template arguments
genesis, genesis_kernels, n_genesis = stage_times(genesis_path, genesis_stage, "demangledName")
cgq_path = Path(args.cgq) if args.cgq else results / f"cgq_{args.scene}.sqlite"
cgq, cgq_kernels, n_cgq = stage_times(cgq_path, cgq_stage, "shortName")
print(f"{args.scene}: kernel ms per step, Genesis ({n_genesis} steps) vs CGQ ({n_cgq} steps)")
print("stage | Genesis | CGQ | ratio | top Genesis kernel | top CGQ kernel")
total_genesis = total_cgq = 0.0
for stage in STAGES:
    g, c = genesis[stage] / n_genesis / 1e6, cgq[stage] / n_cgq / 1e6
    total_genesis += g
    total_cgq += c
    top_g = max(genesis_kernels[stage].items(), key=lambda item: item[1], default=("-", 0))[0]
    top_c = max(cgq_kernels[stage].items(), key=lambda item: item[1], default=("-", 0))[0]
    ratio = f"{g / c:.1f}" if c > 0 else "-"
    print(f"{stage} | {g:.2f} | {c:.2f} | {ratio} | {top_g[:50]} | {top_c[:40]}")
print(f"total | {total_genesis:.2f} | {total_cgq:.2f} | {total_genesis / total_cgq:.1f}")
