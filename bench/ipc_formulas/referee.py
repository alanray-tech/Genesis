"""50-digit exact values of the rows the pass1 and order clauses of compare.py leave, the referee of the parity.

    python bench/ipc_formulas/referee.py --summary acceptance-noreferee

Reads the failing rows of results/summary_<summary>.json (every backend), evaluates the formula of their case on the
float64 inputs in 50-digit arithmetic (mpmath), and writes data/referee.npz with '<case>/<set>/rows' and
'<case>/<set>/exact' (NaN for the discrete outputs). The edge-edge barrier cases (barrier_ee, barrier_ee_unprefixed) are
evaluated, plain or mollified as CGQ's production dispatch decides (its flag and mollifier test, which the row must
already match); a failing row of another case is reported and left without referee.
"""

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from cases import CASES, HESSIAN_CLAMP_RATIO, MOLLIFIER_RATIO

HERE = Path(__file__).resolve().parent
mp.mp.dps = 50
FD_STEP = mp.mpf("1e-25")


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def dot(a, b):
    return sum(s * t for s, t in zip(a, b))


def sub(a, b):
    return [s - t for s, t in zip(a, b)]


def gradient(f, xs):
    """Gradient over the 12 coordinates by central differences, exact to ~FD_STEP^2 at 50 digits."""
    out = []
    for i in range(12):
        plus = [list(x) for x in xs]
        minus = [list(x) for x in xs]
        plus[i // 3][i % 3] += FD_STEP
        minus[i // 3][i % 3] -= FD_STEP
        out.append((f(plus) - f(minus)) / (2 * FD_STEP))
    return mp.matrix(out)


def flagged_distance2(flag):
    pc = bin(flag).count("1")

    def distance2(x):
        if pc == 4:
            n = cross(sub(x[1], x[0]), sub(x[3], x[2]))
            s = dot(sub(x[2], x[0]), n)
            return s * s / dot(n, n)
        if pc == 3:
            if (flag & 3) != 3:
                p, e0, e1 = x[0] if flag & 1 else x[1], x[2], x[3]
            else:
                p, e0, e1 = x[2] if flag & 4 else x[3], x[0], x[1]
            c = cross(sub(e0, p), sub(e1, p))
            return dot(c, c) / dot(sub(e1, e0), sub(e1, e0))
        a, b = x[0] if flag & 1 else x[1], x[2] if flag & 4 else x[3]
        return dot(sub(a, b), sub(a, b))

    return distance2


def cross_norm2(x):
    c = cross(sub(x[1], x[0]), sub(x[3], x[2]))
    return dot(c, c)


def curvature_eigenvalue(r, kappa):
    """8 kappa (r A^2 + (r^2 - 1) ln r / r + (r - 1) ln r A / 2), A = ln r + (r - 1) / r, r clamped from below."""
    r = max(r, mp.mpf(HESSIAN_CLAMP_RATIO))
    log_r = mp.log(r)
    log_term = log_r + (r - 1) / r
    return 8 * kappa * (r * log_term**2 + (r * r - 1) * log_r / r + (r - 1) * log_r * log_term / 2)


def barrier_ee_exact(row, flag, is_mollified):
    xs = [[mp.mpf(float(v)) for v in row[3 * k : 3 * k + 3]] for k in range(4)]
    rest = [[mp.mpf(float(v)) for v in row[12 + 3 * k : 15 + 3 * k]] for k in range(4)]
    d_hat, xi, kappa = (mp.mpf(float(v)) for v in row[24:27])
    distance2 = flagged_distance2(flag)
    D = distance2(xs)
    gD = gradient(distance2, xs)
    dH = d_hat * d_hat + 2 * d_hat * xi
    dt = D - xi * xi
    r = dt / dH
    log_r = mp.log(r)
    log_term = log_r + (r - 1) / r
    clamped = r if dt >= mp.mpf(HESSIAN_CLAMP_RATIO) * dH else mp.mpf(HESSIAN_CLAMP_RATIO)
    if not is_mollified:
        first = 2 * kappa * (r - 1) * log_r * log_term / dH
        g = first * gD
        H = curvature_eigenvalue(clamped, kappa) / (4 * dt * dH) * gD * gD.T
        return g, H
    I1 = cross_norm2(xs)
    gI = gradient(cross_norm2, xs)
    eps = mp.mpf(MOLLIFIER_RATIO) * dot(sub(rest[0], rest[1]), sub(rest[0], rest[1])) * dot(
        sub(rest[2], rest[3]), sub(rest[2], rest[3])
    )
    c = mp.sqrt(I1)
    ratio_sqrt = mp.sqrt(dt) / mp.sqrt(dH)
    mollifier = I1 * (2 * eps - I1) / eps**2
    stress_cross = 4 * kappa * (eps - I1) / eps**2 * (r - 1) ** 2 * log_r**2
    stress_distance = 4 * kappa * mollifier * (r - 1) * log_r * log_term
    c4 = gI / (2 * c)
    c8 = gD / (2 * mp.sqrt(dH) * mp.sqrt(dt))
    g = stress_cross * c * c4 + stress_distance * ratio_sqrt * c8
    l10 = 4 * kappa * (eps - 3 * I1) / eps**2 * (r - 1) ** 2 * log_r**2
    l20 = mollifier * curvature_eigenvalue(clamped, kappa)
    lg = 16 * kappa * (eps - I1) * (r - 1) * log_r * log_term * c * ratio_sqrt / eps**2
    eigenvalues, Q = mp.eigsy(mp.matrix([[l10, lg], [lg, l20]]))
    B = Q * mp.diag([max(e, 0) for e in eigenvalues]) * Q.T
    H = B[0, 0] * c4 * c4.T + B[0, 1] * (c4 * c8.T + c8 * c4.T) + B[1, 1] * c8 * c8.T
    return g, H


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default="acceptance-noreferee")
    parser.add_argument("--reference", default="fmad-true")
    args = parser.parse_args()
    summary = json.loads((HERE / "results" / f"summary_{args.summary}.json").read_text())
    failing = {}
    for results in summary.values():
        for key, result in results.items():
            failing.setdefault(key, set()).update(result["failed_rows"])
    arrays = {}
    with np.load(HERE / "data" / "inputs.npz") as inputs, np.load(
        HERE / "data" / f"out_cgq_{args.reference}.npz"
    ) as reference:
        for key, rows in sorted(failing.items()):
            if not rows:
                continue
            case = CASES[key.split("/")[0]]
            if case.name not in ("barrier_ee", "barrier_ee_unprefixed"):
                print(f"{key}: {len(rows)} failing rows, no referee for this case")
                continue
            rows = np.array(sorted(rows))
            exact = np.full((len(rows), sum(o.width for o in case.outputs)), np.nan)
            for i, i_r in enumerate(rows):
                flag, is_mollified = int(reference[key][i_r, 0]), bool(reference[key][i_r, 1])
                g, H = barrier_ee_exact(inputs[key][i_r], flag, is_mollified)
                exact[i, 2:14] = [float(v) for v in g]
                exact[i, 14:] = [float(H[a, b]) for a in range(12) for b in range(12)]
            arrays[f"{key}/rows"] = rows
            arrays[f"{key}/exact"] = exact
            print(f"{key}: {len(rows)} rows evaluated")
    np.savez(HERE / "data" / "referee.npz", **arrays)
    print(f"wrote {HERE / 'data' / 'referee.npz'}")


if __name__ == "__main__":
    main()
