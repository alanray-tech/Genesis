"""List every offloaded task of the Quadrants kernels whose name contains a pattern, in task order, with its time per
launch and its launch configuration, from the SQLite export of an nsys report.

    python kernel_tasks.py results/genesis_crossed_drop.sqlite kernel_line_search_trial kernel_assemble_rows
"""

import argparse
import collections
import re
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("sqlite")
parser.add_argument("patterns", nargs="+")
args = parser.parse_args()
db = sqlite3.connect(args.sqlite)
strings = dict(db.execute("SELECT id, value FROM StringIds"))
tasks = collections.defaultdict(lambda: [0, 0, set(), 0])
for start, end, name_id, grid_x, block_x, registers, local_mem in db.execute(
    "SELECT start, end, demangledName, gridX, blockX, registersPerThread, localMemoryPerThread "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL"
):
    name = strings.get(name_id, str(name_id))
    if any(pattern in name for pattern in args.patterns):
        entry = tasks[name]
        entry[0] += end - start
        entry[1] += 1
        entry[2].add((grid_x, block_x))
        entry[3] = (registers, local_mem)


def task_index(name: str) -> int:
    match = re.search(r"_kernel_(\d+)_", name)
    return int(match.group(1)) if match else -1


for name, (total, count, configs, (registers, local_mem)) in sorted(tasks.items(), key=lambda item: task_index(item[0])):
    print(
        f"{total / count / 1e3:9.1f} us x {count:5d}  regs {registers:3d} local {local_mem:5d} B  "
        f"{sorted(configs)[:2]}  {name[:120]}"
    )
