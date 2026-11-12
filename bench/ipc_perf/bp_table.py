"""Broad-phase kernel time per step, per Newton iteration and per search, and wall time per step, of Genesis trees and
CGQ on one scene, from the outputs of genesis_profile.py / cgq_profile.py (wall and nsys modes) and the nsys SQLite
exports of the nsys windows.

    python bp_table.py --dir <results dir> --scene crossed_drop --sides before after

Genesis side <tag> reads genesis_<scene>_wall_<tag>.json, genesis_<scene>_nsys_<tag>.json and genesis_<scene>_<tag>.sqlite,
CGQ reads cgq_<scene>_wall.json, cgq_<scene>_nsys.json and cgq_<scene>.sqlite.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

GENESIS_BROAD_PHASE = r"^kernel_contact_(boxes|query|filter)|^build_bvh|^kernel_ef_"
# One launch of the filter per search: its first task
GENESIS_SEARCH = r"^kernel_contact_filter_c\d+_\d+_kernel_0_"
CGQ_BROAD_PHASE = r"^(ef_|bvh_|halfplane_query)"

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=Path, required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--sides", nargs="+", default=["before", "after"])
args = parser.parse_args()


def kernel_times(path: Path, name_column: str, pattern: str, search_pattern: str | None) -> tuple[float, int, int]:
    """The time of the kernels matching `pattern` within the steps of the window, the launches matching
    `search_pattern`, and the step count."""
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    steps = list(db.execute("SELECT start, end FROM NVTX_EVENTS WHERE text LIKE 'step %' ORDER BY start"))
    total, n_searches = 0, 0
    for k_start, k_end, name_id in db.execute(f"SELECT start, end, {name_column} FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if any(start <= k_start <= end for start, end in steps):
            name = strings.get(name_id, str(name_id))
            if re.search(pattern, name):
                total += k_end - k_start
            if search_pattern is not None and re.search(search_pattern, name):
                n_searches += 1
    return total, n_searches, len(steps)


print(f"{args.scene}: side | wall ms/step (median) | Newton/step | broad phase ms/step | ms/Newton iteration | "
      "searches/step | ms/search")
sides = [(side, f"genesis_{args.scene}_{{}}_{side}", f"genesis_{args.scene}_{side}.sqlite") for side in args.sides]
sides.append(("CGQ", f"cgq_{args.scene}_{{}}", f"cgq_{args.scene}.sqlite"))
for side, json_name, sqlite_name in sides:
    wall = json.loads((args.dir / f"{json_name.format('wall')}.json").read_text())["summary"]
    nsys = json.loads((args.dir / f"{json_name.format('nsys')}.json").read_text())
    is_cgq = side == "CGQ"
    total, n_searches, n_steps = kernel_times(
        args.dir / sqlite_name,
        "shortName" if is_cgq else "demangledName",
        CGQ_BROAD_PHASE if is_cgq else GENESIS_BROAD_PHASE,
        None if is_cgq else GENESIS_SEARCH,
    )
    n_newton = sum(row["newton_iterations"] for row in nsys["rows"])
    ms = total / 1e6
    searches = f"{n_searches / n_steps:.1f} | {ms / max(n_searches, 1):.3f}" if not is_cgq else "- | -"
    print(
        f"{side} | {wall['step_ms_median']:.2f} | {n_newton / n_steps:.1f} | {ms / n_steps:.3f} | "
        f"{ms / max(n_newton, 1):.3f} | {searches}"
    )
