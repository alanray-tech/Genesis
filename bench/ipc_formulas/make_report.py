"""Per-function clause counts of the acceptance comparison, with example inputs, for REPORT.md.

    python bench/ipc_formulas/make_report.py [--label acceptance]

Reads results/summary_<label>.json and data/inputs.npz, writes results/clauses_<label>.md: per case (all input sets),
the rows, the rows passing by pass1, by order and by referee (compare.py), the largest |g - c| / s of the last two, the
failing rows, and for the order and referee clauses the 3 rows of largest error with their inputs. One table per
backend.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from cases import CASES

HERE = Path(__file__).resolve().parent
N_EXAMPLES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="acceptance")
    args = parser.parse_args()
    summary = json.loads((HERE / "results" / f"summary_{args.label}.json").read_text())
    lines = [f"# Clauses of the parity rule per function: {args.label}", ""]
    with np.load(HERE / "data" / "inputs.npz") as inputs:
        for run, results in summary.items():
            lines += [f"## {run}", ""]
            lines.append("| case | function | rows | pass1 | order | order max err / s | referee | referee max err / s "
                         "| failures |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            examples = {}
            for case_name in CASES:
                keys = [k for k in results if k.split("/")[0] == case_name]
                if not keys:
                    continue
                rows = sum(results[k]["rows"] for k in keys)
                failures = sum(results[k]["failures"] for k in keys)
                cells = []
                for clause in ("order", "referee"):
                    n = sum(results[k][clause]["rows"] for k in keys)
                    error = max(results[k][clause]["max_error"] for k in keys)
                    cells += [str(n), f"{error:.2e}" if n else "-"]
                    ranked = sorted(
                        ((results[k][clause]["max_error"], k, i) for k in keys for i in results[k][clause]["examples"]),
                        reverse=True,
                    )
                    if ranked:
                        examples[(case_name, clause)] = ranked[:N_EXAMPLES]
                n_order = sum(results[k]["order"]["rows"] for k in keys)
                n_referee = sum(results[k]["referee"]["rows"] for k in keys)
                pass1 = rows - failures - n_order - n_referee
                lines.append(
                    f"| {case_name} | {CASES[case_name].function} | {rows} | {pass1} | {cells[0]} | {cells[1]} | "
                    f"{cells[2]} | {cells[3]} | {failures} |"
                )
            lines += ["", f"Example inputs ({run}): set, row, input row (cases.py documents the columns)", ""]
            for (case_name, clause), ranked in examples.items():
                for _, key, i_r in ranked:
                    values = ", ".join(f"{v:.17g}" for v in inputs[key][i_r])
                    lines.append(f"- {case_name} / {clause}: {key.split('/', 1)[1]} row {i_r}: [{values}]")
            lines.append("")
    path = HERE / "results" / f"clauses_{args.label}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
