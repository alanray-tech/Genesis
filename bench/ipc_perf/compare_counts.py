"""Compare the solver work per step of Genesis (genesis_profile.py --mode count) and CGQ (cgq_profile.py) on the same
frames: Newton iterations, line-search trials, PCG iterations, and the Genesis kernel calls and readbacks.

    python compare_counts.py crossed_drop [--start 20] [--end 200]
"""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("scene")
parser.add_argument("--start", type=int, default=20)
parser.add_argument("--end", type=int, default=10**9)
parser.add_argument("--cgq-mode", default="wall")
args = parser.parse_args()
results = Path(__file__).parent / "results"
sides = {
    "genesis": json.loads((results / f"genesis_{args.scene}_count.json").read_text())["rows"],
    "cgq": json.loads((results / f"cgq_{args.scene}_{args.cgq_mode}.json").read_text())["rows"],
}
frames = set.intersection(*({row["frame"] for row in rows} for rows in sides.values()))
frames = {frame for frame in frames if args.start <= frame < args.end}
iterations = {}
for side, rows in sides.items():
    rows = [row for row in rows if row["frame"] in frames]
    it = np.array([row["newton_iterations"] for row in rows])
    iterations[side] = it
    ls = sum(row["ls_trials"] for row in rows)
    pcg = sum(row["pcg_iterations"] for row in rows)
    line = (
        f"{side:8s} frames {len(rows)}: Newton {it.sum()} (median {np.median(it):g}, max {it.max()}), "
        f"trials per Newton {ls / it.sum():.2f}, PCG per Newton {pcg / it.sum():.1f}, PCG per step {pcg / len(rows):.0f}"
    )
    if side == "genesis":
        calls = sum(row["kernel_calls"] for row in rows)
        readbacks = sum(row["readbacks"] - row["readback_sites"].get("misc.py:1027 qd_to_numpy", 0) for row in rows)
        line += (
            f", kernel calls per step {calls / len(rows):.0f} ({calls / it.sum():.1f} per Newton), readbacks per step "
            f"{readbacks / len(rows):.1f} ({readbacks / it.sum():.2f} per Newton), pairs max "
            f"{max(row['n_pairs'][0] for row in rows)} PT / {max(row['n_pairs'][1] for row in rows)} EE"
        )
    print(line)
difference = iterations["genesis"] - iterations["cgq"]
print(f"Newton per frame, Genesis - CGQ: mean {difference.mean():.2f}, frames above +2: {(difference > 2).sum()}")
