"""Print the per-step nsys summaries (nsys_summary.py) of a scene side by side, Genesis then CGQ.

    python print_summary.py crossed_drop [--top 14]
"""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("scene")
parser.add_argument("--top", type=int, default=14)
parser.add_argument("--per-step", action="store_true")
args = parser.parse_args()
results = Path(__file__).parent / "results"
for side in ("genesis", "cgq"):
    data = json.loads((results / f"{side}_{args.scene}.summary.json").read_text())
    summary = data["summary"]
    if args.per_step:
        for step in data["per_step"]:
            print(
                f"  {side} {step['step']:9s} wall {step['wall_ms']:6.1f} ms, GPU busy {step['gpu_busy_ms']:6.1f} ms, "
                f"{step['kernels']:5d} kernels, {step['memcpy']:4d} memcpy, Newton marks {step['newton_iterations_marked']}"
            )
    print(f"===== {side}: {summary['steps']} steps")
    print(" ", {key: round(value, 2) for key, value in summary.items() if isinstance(value, float)})
    print("  API calls per step", {key: round(value, 1) for key, value in summary["api_calls_per_step"].items()})
    print("  API ms per step", {key: round(value, 2) for key, value in summary["api_ms_per_step"].items()})
    for kernel in summary["top_kernels"][: args.top]:
        print(
            f"  {kernel['ms_per_step']:6.3f} ms {kernel['launches_per_step']:7.1f} x {kernel['avg_us']:7.1f} us "
            f"{kernel['name'][:64]:64s} {kernel['grid_x_block'][:2]}"
        )
