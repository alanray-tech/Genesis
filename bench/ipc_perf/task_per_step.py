"""Print, for every NVTX step of an nsys SQLite export, the durations of the launches of the kernels whose full name
contains a pattern, in launch order (one per Newton iteration or line-search trial).

    python task_per_step.py results/genesis_crossed_drop.sqlite kernel_line_search_trial_c604_0_kernel_10_
"""

import argparse
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("sqlite")
parser.add_argument("pattern")
args = parser.parse_args()
db = sqlite3.connect(args.sqlite)
strings = dict(db.execute("SELECT id, value FROM StringIds"))
launches = sorted(
    (start, end)
    for start, end, name_id in db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
    if args.pattern in strings.get(name_id, "")
)
for start, end, text in db.execute("SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'step %' ORDER BY start"):
    durations = [(k_end - k_start) / 1e3 for k_start, k_end in launches if start <= k_start <= end]
    print(f"{text:9s}: " + " ".join(f"{us:.0f}" for us in durations))
