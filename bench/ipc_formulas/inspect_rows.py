"""Print the inputs and both outputs of the failing rows of a case and set, for debugging the parity.

    python bench/ipc_formulas/inspect_rows.py <case>/<set> [--run cuda_fast-math-false] [--reference fmad-false] [-n 5]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cases import CASES

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("--run", default="cuda_fast-math-false")
    parser.add_argument("--reference", default="fmad-true")
    parser.add_argument("--label", default="acceptance")
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args()
    np.set_printoptions(precision=17, linewidth=200)
    summary = json.loads((HERE / "results" / f"summary_{args.label}.json").read_text())
    result = summary[args.run][args.key]
    print(json.dumps({k: v for k, v in result["outputs"].items()}, indent=1))
    case = CASES[args.key.split("/")[0]]
    with np.load(HERE / "data" / "inputs.npz") as i, np.load(HERE / "data" / f"out_cgq_{args.reference}.npz") as c:
        with np.load(HERE / "data" / f"out_genesis_{args.run}.npz") as g:
            rows, ref, gen = i[args.key], c[args.key], g[args.key]
    for i_r in result["failed_rows"][: args.n]:
        print(f"--- row {i_r}\ninput {rows[i_r]!r}")
        col = 0
        for output in case.outputs:
            r, q = ref[i_r, col : col + output.width], gen[i_r, col : col + output.width]
            if not np.array_equal(r, q, equal_nan=True):
                print(f"{output.name}\n  cgq     {r!r}\n  genesis {q!r}")
            col += output.width


if __name__ == "__main__":
    main()
