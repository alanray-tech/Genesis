"""Compare the outputs of the Genesis side with the CGQ reference, per case, input set and backend.

    python bench/ipc_formulas/compare.py --reference fmad-true --ieee fmad-false --order _sum-order-eigen \
        --referee data/referee.npz --label acceptance --runs cuda_fast-math-false cpu_fast-math-false

The reference c is CGQ as production builds it (nvcc default, --fmad=true), c_ieee the IEEE build (--fmad=false), s
the largest magnitude of an output (all its components) over the input set. For a candidate x, pass1(x) holds on a row
when every real output has max |x - c| <= max(1e-12 * s, 2 * max |c_ieee - c|) over its components and every discrete
output (flag, branch, active bit, verdict, status, iteration count) equals c or c_ieee. A row of Genesis g passes by:
1. pass1: pass1(g);
2. order: pass1(g_order), g_order being the same Genesis code run with Eigen's summation order of 3-term dot products
   (the order of CGQ's Eigen-based code), so that g differs from CGQ by its summation order alone;
3. referee: every discrete output passes for g or g_order, and every real output passes for g or g_order or satisfies
   max |g - e| <= max(max |c - e|, 1e-12 * s), e its 50-digit exact value (referee.py).
Without '--ieee' and '--order' the rule reduces to 1e-12 * s and equality with c. Two NaN (or two equal infinities)
agree. Writes results/table_<label>.md and results/summary_<label>.json, which lists every failing row and, per case,
the rows each clause was needed for.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cases import CASES

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-12
N_EXAMPLES = 3
CLAUSES = ("pass1", "order", "referee")


def elementwise_error(a, b):
    """|a - b|, zero where both are NaN or equal infinities, infinite where only one is NaN."""
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    with np.errstate(invalid="ignore"):
        error = np.where(same, 0.0, np.abs(a - b))
    return np.where(np.isnan(error), np.inf, error)


def pass1(x, reference, ieee, scale, is_discrete):
    """Rows where the candidate x agrees with CGQ within CGQ's own rounding spread (see the module docstring)."""
    if is_discrete:
        is_pass = (elementwise_error(x, reference) == 0.0).all(axis=1)
        if ieee is not None:
            is_pass |= (elementwise_error(x, ieee) == 0.0).all(axis=1)
        return is_pass
    bound = np.full(len(x), TOLERANCE * scale)
    if ieee is not None:
        bound = np.maximum(bound, 2.0 * elementwise_error(ieee, reference).max(axis=1))
    return elementwise_error(x, reference).max(axis=1) <= bound


def compare_case(case, gen, ref, ieee, order, exact):
    n = len(ref)
    is_pass1 = np.ones(n, dtype=bool)
    is_order = np.ones(n, dtype=bool) if order is not None else np.zeros(n, dtype=bool)
    is_mixed = np.ones(n, dtype=bool)
    error = np.zeros(n)
    outputs = {}
    col = 0
    for output in case.outputs:
        columns = slice(col, col + output.width)
        g, c = gen[:, columns], ref[:, columns]
        i = None if ieee is None else ieee[:, columns]
        finite = np.isfinite(c)
        scale = float(np.abs(c[finite]).max()) if finite.any() else 0.0
        g_pass = pass1(g, c, i, scale, output.is_discrete)
        o_pass = pass1(order[:, columns], c, i, scale, output.is_discrete) if order is not None else np.zeros(n, bool)
        r_pass = np.zeros(n, dtype=bool)
        stats = {"is_discrete": output.is_discrete, "pass1_fail": int((~g_pass).sum())}
        if not output.is_discrete:
            row_error = elementwise_error(g, c).max(axis=1)
            normalized = row_error / scale if scale > 0 else row_error
            error = np.maximum(error, normalized)
            stats.update(scale=scale, max_error=float(normalized.max()) if n else 0.0)
            if exact is not None:
                e = exact[:, columns]
                has_exact = ~np.isnan(e).all(axis=1)
                r_pass = has_exact & (
                    elementwise_error(g, e).max(axis=1)
                    <= np.maximum(elementwise_error(c, e).max(axis=1), TOLERANCE * scale)
                )
        is_pass1 &= g_pass
        is_order &= o_pass
        is_mixed &= g_pass | o_pass | r_pass
        stats["fail"] = int((~(g_pass | o_pass | r_pass)).sum())
        outputs[output.name] = stats
        col += output.width
    clause = np.full(n, "fail", dtype=object)
    clause[is_mixed] = "referee"
    clause[is_order] = "order"
    clause[is_pass1] = "pass1"
    return clause, error, outputs


def summarize(clause, error, name):
    rows = np.flatnonzero(clause == name)
    examples = rows[np.argsort(-error[rows])][:N_EXAMPLES]
    return {"rows": int(len(rows)), "max_error": float(error[rows].max()) if len(rows) else 0.0,
            "examples": examples.tolist()}


def compare_run(reference, ieee, order, genesis, referee):
    results = {}
    for key in sorted(reference.files):
        if key not in genesis.files:
            continue
        case = CASES[key.split("/")[0]]
        ref = reference[key]
        exact = None
        if referee is not None and f"{key}/rows" in referee.files:
            exact = np.full_like(ref, np.nan)
            exact[referee[f"{key}/rows"]] = referee[f"{key}/exact"]
        clause, error, outputs = compare_case(
            case,
            genesis[key],
            ref,
            None if ieee is None else ieee[key],
            None if order is None or key not in order.files else order[key],
            exact,
        )
        results[key] = {
            "rows": len(ref),
            "failures": int((clause == "fail").sum()),
            "failed_rows": np.flatnonzero(clause == "fail").tolist(),
            "max_error": float(error.max()) if len(error) else 0.0,
            **{name: summarize(clause, error, name) for name in CLAUSES[1:]},
            "outputs": outputs,
        }
    return results


def format_table(label, runs):
    lines = [f"# IPC formula parity: {label}", ""]
    lines.append(
        "Per case and input set: rows, then per backend the failing rows, the largest real error over the scale of its "
        "output (max |g - c| / s), and the rows that passed by the order clause and by the referee (compare.py)."
    )
    lines.append("")
    run_names = list(runs)
    columns = ("failures", "max err / s", "order rows", "referee rows")
    header = "| case | function | set | rows | " + " | ".join(f"{name} {c}" for name in run_names for c in columns)
    lines += [header + " |", "|" + "---|" * (4 + len(columns) * len(run_names))]
    for key in sorted(set().union(*[set(r) for r in runs.values()])):
        case_name, set_name = key.split("/", 1)
        first = next(r[key] for r in runs.values() if key in r)
        cells = [case_name, CASES[case_name].function, set_name, str(first["rows"])]
        for name in run_names:
            result = runs[name].get(key)
            if result is None:
                cells += ["-"] * len(columns)
                continue
            has_real = any(not o["is_discrete"] for o in result["outputs"].values())
            cells += [
                str(result["failures"]),
                f"{result['max_error']:.2e}" if has_real else "-",
                str(result["order"]["rows"]),
                str(result["referee"]["rows"]),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    totals = {name: sum(r["failures"] for r in run.values()) for name, run in runs.items()}
    rows = {name: sum(r["rows"] for r in run.values()) for name, run in runs.items()}
    lines += ["", "Totals: " + ", ".join(f"{name}: {totals[name]} failing rows of {rows[name]}" for name in run_names)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="fmad-true", help="suffix of data/out_cgq_<reference>.npz")
    parser.add_argument("--ieee", default=None, help="suffix of the IEEE CGQ build, data/out_cgq_<ieee>.npz")
    parser.add_argument("--order", default=None, help="suffix of the Eigen-order Genesis run of each backend")
    parser.add_argument("--referee", type=Path, default=None, help="exact values of referee.py")
    parser.add_argument("--runs", nargs="+", required=True, help="suffixes of data/out_genesis_<run>.npz")
    parser.add_argument("--reference-run", action="store_true", help="the runs are CGQ outputs (out_cgq_<run>.npz)")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    label = args.label or f"cgq-{args.reference}_vs_" + "+".join(args.runs)
    data = HERE / "data"
    referee = np.load(args.referee) if args.referee is not None and args.referee.is_file() else None
    ieee = np.load(data / f"out_cgq_{args.ieee}.npz") if args.ieee else None
    runs = {}
    with np.load(data / f"out_cgq_{args.reference}.npz") as reference:
        for run in args.runs:
            order = np.load(data / f"out_genesis_{run}{args.order}.npz") if args.order else None
            path = data / (f"out_cgq_{run}.npz" if args.reference_run else f"out_genesis_{run}.npz")
            with np.load(path) as genesis:
                runs[run] = compare_run(reference, ieee, order, genesis, referee)
            if order is not None:
                order.close()
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    table = format_table(label, runs)
    (results_dir / f"table_{label}.md").write_text(table, encoding="utf-8")
    (results_dir / f"summary_{label}.json").write_text(json.dumps(runs, indent=1), encoding="utf-8")
    print(table)
    is_pass = True
    for run, results in runs.items():
        failing = {k: v["failures"] for k, v in results.items() if v["failures"]}
        is_pass = is_pass and not failing
        print(f"{run}: {'PASS' if not failing else 'FAIL'}" + (f" {failing}" if failing else ""))
    sys.exit(0 if is_pass else 1)


if __name__ == "__main__":
    main()
