"""Print the static stats of sass_stats.py (results/sass_<side>.json) for the functions matching a regex, one per line.

    python print_sass.py genesis "kernel_assemble_rows"
"""

import argparse
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("side")
parser.add_argument("match", nargs="?", default=".")
parser.add_argument("--min-instructions", type=int, default=200)
args = parser.parse_args()
rows = json.loads((Path(__file__).parent / "results" / f"sass_{args.side}.json").read_text())
keys = ("regs", "stack", "spill_stores", "spill_loads", "local", "instructions", "fp64", "local_loads", "local_stores",
        "global_atomics", "branches")
print("function | " + " | ".join(keys))
for name, entry in sorted(rows.items()):
    if re.search(args.match, name) and entry.get("instructions", 0) >= args.min_instructions:
        print(f"{name[:80]} | " + " | ".join(str(entry.get(key, "-")) for key in keys))
