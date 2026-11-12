"""Print the step time and Newton iterations of every frame of a scene, Genesis next to CGQ, from their wall runs.

    python frame_times.py cloth_on_box [--start 40] [--end 60] [--tag _tol0.05]
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("scene")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=10**9)
parser.add_argument("--tag", default="")
args = parser.parse_args()
results = Path(__file__).parent / "results"
rows = {
    side: {row["frame"]: row for row in json.loads((results / f"{side}_{args.scene}_wall{args.tag}.json").read_text())["rows"]}
    for side in ("genesis", "cgq")
}
for frame in sorted(rows["genesis"].keys() & rows["cgq"].keys()):
    if args.start <= frame < args.end:
        genesis, cgq = rows["genesis"][frame], rows["cgq"][frame]
        print(
            f"frame {frame:4d}: Genesis {genesis['step_ms']:7.1f} ms {genesis['newton_iterations']:4d} it | "
            f"CGQ {cgq['step_ms']:6.1f} ms {cgq['newton_iterations']:4d} it"
        )
