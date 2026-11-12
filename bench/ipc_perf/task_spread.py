"""Print the spread of the duration of every launch of the kernels whose name contains a pattern, from the SQLite export
of an nsys report: a duration that stays flat while the pair count changes points at the latency of one thread.

    python task_spread.py results/genesis_crossed_drop.sqlite kernel_line_search_trial_c604_0_kernel_10_
"""

import argparse
import sqlite3

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("sqlite")
parser.add_argument("patterns", nargs="+")
args = parser.parse_args()
db = sqlite3.connect(args.sqlite)
strings = dict(db.execute("SELECT id, value FROM StringIds"))
for pattern in args.patterns:
    durations = np.array(
        [
            end - start
            for start, end, name_id in db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
            if pattern in strings.get(name_id, "")
        ]
    )
    quantiles = np.percentile(durations, [0, 10, 50, 90, 100]) / 1e3
    print(f"{pattern}: {len(durations)} launches, us min/p10/median/p90/max " + " / ".join(f"{q:.0f}" for q in quantiles))
