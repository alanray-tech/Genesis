"""List every kernel of the NVTX step window of an nsys SQLite export with its time and launches per step.

    python kernel_list.py results/cgq_crossed_drop.sqlite [--min-us 1]
"""

import argparse
import collections
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("sqlite")
parser.add_argument("--min-us", type=float, default=1.0, help="hide kernels below this time per step")
args = parser.parse_args()
db = sqlite3.connect(args.sqlite)
strings = dict(db.execute("SELECT id, value FROM StringIds"))
steps = list(db.execute("SELECT start, end FROM NVTX_EVENTS WHERE text LIKE 'step %' ORDER BY start"))
start, end = steps[0][0], steps[-1][1]
totals = collections.defaultdict(lambda: [0, 0])
for k_start, k_end, name_id in db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
    if start <= k_start <= end:
        entry = totals[strings.get(name_id, str(name_id))]
        entry[0] += k_end - k_start
        entry[1] += 1
for name, (total, count) in sorted(totals.items(), key=lambda item: -item[1][0]):
    us_per_step = total / len(steps) / 1e3
    if us_per_step >= args.min_us:
        print(f"{us_per_step:9.1f} us/step {count / len(steps):7.1f} launches/step  {name}")
